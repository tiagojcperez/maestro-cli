```
  ██   ██  ████  ██████  ████ ██████ █████   ████
  ███ ███ ██  ██ ██     ██      ██   ██  ██ ██  ██
  ██ █ ██ ██████ █████   ███    ██   █████  ██  ██
  ██   ██ ██  ██ ██        ██   ██   ██ ██  ██  ██
  ██   ██ ██  ██ ██████ ████    ██   ██  ██  ████

  Maestro CLI -- Version 2.4.0
  CLI orchestrator for multi-step AI execution plans
```

[![CI](https://github.com/tiagojcperez/maestro-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/tiagojcperez/maestro-cli/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Maestro turns a YAML file into a parallel, dependency-aware pipeline of AI agents and shell commands.**
You declare tasks and how they depend on each other; Maestro schedules them as a DAG across the engines you
choose, passes context between steps, enforces cost budgets and quality gates, and records every run
deterministically -- all from one dependency-light CLI (PyYAML is the only required dependency, everything
else is stdlib or optional).

```yaml
# the smallest useful plan
version: 1
name: hello
tasks:
  - id: greet
    engine: claude
    prompt: "Say hello in three languages."
```

```bash
maestro run plan.yaml      # runs the DAG; one engine call, deterministic logs in .maestro-runs/
```

### Supported Engines

<p>
<img src="https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=claude&logoColor=white" alt="Claude"> <img src="https://img.shields.io/badge/Codex-412991?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0yMi4yODIgOS44MjFhNS45ODUgNS45ODUgMCAwIDAtLjUxNi00LjkxIDYuMDQ2IDYuMDQ2IDAgMCAwLTYuNTEtMi45QTYuMDY1IDYuMDY1IDAgMCAwIDQuOTgxIDQuMThhNS45ODUgNS45ODUgMCAwIDAtMy45OTggMi45IDYuMDQ2IDYuMDQ2IDAgMCAwIC43NDMgNy4wOTcgNS45OCA1Ljk4IDAgMCAwIC41MSA0LjkxMSA2LjA1MSA2LjA1MSAwIDAgMCA2LjUxNSAyLjlBNS45ODUgNS45ODUgMCAwIDAgMTMuMjYgMjRhNi4wNTYgNi4wNTYgMCAwIDAgNS43NzItNC4yMDYgNS45OSA1Ljk5IDAgMCAwIDMuOTk3LTIuOSA2LjA1NiA2LjA1NiAwIDAgMC0uNzQ3LTcuMDczek0xMy4yNiAyMi40M2E0LjQ3NiA0LjQ3NiAwIDAgMS0yLjg3Ni0xLjA0bC4xNDEtLjA4MSA0Ljc3OS0yLjc1OGEuNzk1Ljc5NSAwIDAgMCAuMzkyLS42ODF2LTYuNzM3bDIuMDIgMS4xNjhhLjA3MS4wNzEgMCAwIDEgLjAzOC4wNTJ2NS41ODNhNC41MDQgNC41MDQgMCAwIDEtNC40OTQgNC40OTR6TTMuNiAxOC4zMDRhNC40NyA0LjQ3IDAgMCAxLS41MzUtMy4wMTRsLjE0Mi4wODUgNC43ODMgMi43NTlhLjc3MS43NzEgMCAwIDAgLjc4IDBsNS44NDMtMy4zNjl2Mi4zMzJhLjA4LjA4IDAgMCAxLS4wMzMuMDYyTDkuNzQgMTkuOTVhNC41IDQuNSAwIDAgMS02LjE0LTEuNjQ2ek0yLjM0IDcuODk2YTQuNDg1IDQuNDg1IDAgMCAxIDIuMzY2LTEuOTczVjExLjZhLjc2Ni43NjYgMCAwIDAgLjM4OC42NzdsNS44MTUgMy4zNTUtMi4wMiAxLjE2OGEuMDc2LjA3NiAwIDAgMS0uMDcxIDBsLTQuODMtMi43ODZBNC41MDQgNC41MDQgMCAwIDEgMi4zNCA3Ljg3MnptMTYuNTk3IDMuODU1bC01LjgzMy0zLjM4N0wxNS4xMTkgNy4yYS4wNzYuMDc2IDAgMCAxIC4wNzEgMGw0LjgzIDIuNzkxYTQuNDk0IDQuNDk0IDAgMCAxLS42NzYgOC4xMDV2LTUuNjc4YS43OS43OSAwIDAgMC0uNDA3LS42Njd6bTIuMDEtMy4wMjNsLS4xNDEtLjA4NS00Ljc3NC0yLjc4MmEuNzc2Ljc3NiAwIDAgMC0uNzg1IDBMOS40MDkgOS4yM1Y2Ljg5N2EuMDY2LjA2NiAwIDAgMSAuMDI4LS4wNjFsNC44My0yLjc4N2E0LjUgNC41IDAgMCAxIDYuNjggNC42NnptLTEyLjY0IDQuMTM1bC0yLjAyLTEuMTY0YS4wOC4wOCAwIDAgMS0uMDM4LS4wNTdWNi4wNzVhNC41IDQuNSAwIDAgMSA3LjM3NS0zLjQ1M2wtLjE0Mi4wOEw4LjcwNCA1LjQ2YS43OTUuNzk1IDAgMCAwLS4zOTMuNjgxem0xLjA5Ny0yLjM2NWwyLjYwMi0xLjUgMi42MDcgMS41djIuOTk5bC0yLjU5NyAxLjUtMi42MDctMS41eiIvPjwvc3ZnPg==&logoColor=white" alt="Codex"> <img src="https://img.shields.io/badge/Gemini-886FBF?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini"> <img src="https://img.shields.io/badge/Copilot-000?style=for-the-badge&logo=githubcopilot&logoColor=white" alt="Copilot"> <img src="https://img.shields.io/badge/Qwen-5A29E4?style=for-the-badge&logo=alibabadotcom&logoColor=white" alt="Qwen"> <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"> <img src="https://img.shields.io/badge/Llama-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="Llama">
</p>

## Why Maestro?

- **DAG scheduling** -- declare dependencies between tasks, Maestro runs them in the right order with configurable parallelism; matrix tasks for Cartesian product expansion; task groups for nested sub-plans; batch task mode for chunked item processing
- **Engine-agnostic** -- mix `codex exec`, `claude --print`, `gemini`, `copilot`, `qwen`, `ollama`, `llama` (local via llama-cpp), and raw shell commands in the same plan; per-engine model aliases and reasoning effort control
- **Zero framework deps** -- only PyYAML; shells out to engine CLIs (no AI SDK lock-in)
- **Smart context** -- 9 context modes: raw, selective (BM25 chunk-level), summarized, map_reduce, recursive, layered (L0/L1/L2 tiers), structural (package-aware code symbol extraction with re-export resolution and PageRank scoring), council (multi-model deliberation), knowledge_graph (entity extraction); progressive compaction (5-stage pipeline); intent-driven BM25 filtering; priority-based eviction; privacy-aware pipeline (`output_redact`, `context_allowlist`); automatic compression on retry; context retrieval trajectory (`explain --context`)
- **LLM-as-Judge** -- quality gates with typed assertions (`contains`, `regex`, `is-json`, `cost_under`, `duration_under`), Likert-scale rubrics, G-Eval two-phase scoring, adversarial debate judge (`method: debate`), comparative evaluation, named presets, `guard_command` validators, quorum voting (majority/unanimous/any), and timeout auto-scaling
- **Content-addressable caching** -- policy-versioned SHA-256 Merkle DAG hash per task, short-lived negative cache for failures (`negative_cache_ttl_sec`), and contamination-aware cache bypass for untrusted/tainted/partial/tool-failure outputs; `maestro explain` shows cache hit/miss reasons; `maestro status` shows task staleness
- **Cost-aware** -- per-task cost and token tracking, budget limits with warning thresholds, cross-run budget tracking (`budget_period`), per-engine pricing tables, `maestro diff` to compare runs
- **Resilient** -- retries with backoff strategies, error feedback injection, auto-escalation (retry with higher-tier model), cross-engine fallback, checkpoint protocol, handoff reports, resume from failure, declarative runtime policies (block/warn/audit), circuit breakers
- **Adaptive** -- mid-task signals (`signals: true`) for progress reporting, budget queries, timeout extensions; dynamic task decomposition (`dynamic_group: true`) for LLM-generated sub-plans at runtime; prompt-relevant cross-run knowledge auto-injection with lightweight `{{ knowledge_index }}`; MCTS workflow search (`mcts.py`) with draft/debug/improve trichotomy and `tree.jsonl` persistence; self-evolving `maestro replan` with multi-variant search, tournament selection, elitism, diversity floor, novelty/knowledge priors, and stepping-stone continuity
- **Persistent memory** -- SQLite-backed Knowledge + Memory v2 with WAL, automatic JSONL migration, bi-temporal records, provenance/trust labels, retrieval-dominance quarantine, and score history (`plan_hash`, `quality_score`) for future pruning/search
- **Blame attribution** -- `maestro blame` traces failure causality via dependency graph backward walk, classifies root causes, provides confidence scores and suggested fixes
- **Security** -- `context_trust: trusted | untrusted` with transitive taint propagation and injection stripping; control flow integrity (`observation_block`); trajectory-level guardrails (`trajectory_guard`); semantic firewall for MCP metadata (`mcp_servers[].description`, role filtering, untrusted tool-doc reminders, `mcp_servers[].is_concurrency_safe`) plus optional pass-2 classification via top-level `firewall_model`; phantom workspace for destructive commands; 23 security audit rules (SEC001-SEC023); `maestro audit --fix` auto-remediation; cross-phase security contract (machine-generated plans audited before execution, tainted variants blocked, consolidation safety gates)
- **Protocol integration** -- AG-UI event stream (`POST /api/agui/runs`) for any compatible frontend; MCP server (`maestro mcp-server`) exposes 12 tools, 8 resources, 3 prompts via stdio; MCP client (`mcp_servers` + `mcp_tools`) lets tasks use external MCP tool providers; OTLP exporter (`maestro export-otel`) emits `gen_ai.*` attributes, optional content capture/redaction, and task events such as `knowledge_poison_alert` and `memory_write`
- **Production-ready** -- secrets masking, plan imports for DRY composition, tags for task filtering, approval gates for human-in-the-loop, `allowed_tools:` per-task tool restriction for prompt injection containment, semantic model routing (`model: auto`) with adaptive temporal routing and cross-task affinity learning, population-based search (best-of-N models), git worktree isolation per task, skill registry with recommendations/trigger metadata (`maestro skill`), CI failure analysis (`maestro ci-analyze`)
- **Observable** -- Web UI dashboard with collaboration surfaces (owners, blockers, recent activity), JSONL streaming, webhook notifications, HTML reports, event sourcing with tamper detection, `suggest`, `eval`, `report`, `diff`, `doctor`, and more

v1 stability contract: [docs/V1_API_FREEZE.md](docs/V1_API_FREEZE.md) | Migration from 0.x: [docs/MIGRATING_TO_V1.md](docs/MIGRATING_TO_V1.md)

## Python SDK

Maestro CLI ships with a programmatic API and `py.typed` marker for static analysis:

```python
from maestro_cli import load_plan, run_plan

plan = load_plan("plan.yaml")
result = run_plan(plan)
```

## Install

```bash
git clone https://github.com/tiagojcperez/maestro-cli.git
cd maestro-cli

pip install -e .            # CLI only
pip install -e ".[live]"    # CLI + Rich live table (--output live)
pip install -e ".[tui]"     # CLI + Textual TUI (--output tui)
pip install -e ".[web]"     # CLI + Web UI (FastAPI + uvicorn)
pip install -e ".[agui]"    # CLI + AG-UI protocol endpoint
pip install -e ".[mcp]"     # CLI + MCP server for IDE integration
pip install -e ".[otel]"    # CLI + OpenTelemetry OTLP exporter
```

Requires Python >= 3.11 and at least one engine CLI on PATH (`codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`, `llama`).

## Quickstart

```bash
# Validate a plan (examples/demo_plan.yaml runs with no API keys)
maestro validate examples/demo_plan.yaml
# -> "Plan is valid: ... name: demo, tasks: 4"

# Dry run (build commands without executing)
maestro run examples/demo_plan.yaml --dry-run

# Run with parallel tasks
maestro run examples/demo_plan.yaml --max-parallel 3 --execution-profile yolo

# Resume from last failed run
maestro run examples/demo_plan.yaml --resume-last

# Stream structured events to stdout
maestro run examples/demo_plan.yaml --output jsonl

# Run multiple plans with shared budget
maestro run examples/demo_plan.yaml examples/demo_plan.yaml --parallel

# Adaptive re-planning on failure
maestro replan examples/demo_plan.yaml --max-attempts 3
maestro replan examples/demo_plan.yaml --variants 4 --selection-policy ucb1 --exploration-constant 1.4
maestro replan examples/demo_plan.yaml --variants 4 --population-strategy tournament --tournament-size 3
maestro replan examples/demo_plan.yaml --variants 6 --population-strategy tournament --tournament-size 3 --elite-count 2
maestro replan examples/demo_plan.yaml --variants 6 --population-strategy tournament --tournament-size 4 --elite-count 2 --diversity-floor 0.35

# Autonomous metric-driven iteration loop (your plan needs a watch: block)
maestro watch your-plan.yaml --output tui

# Multi-model interactive chat
maestro chat --engine claude --model sonnet

# Trace failure causality in a completed run
maestro blame .maestro-runs/<run-dir>/

# Diagnose environment
maestro doctor --json

# Security audit a plan
maestro audit examples/demo_plan.yaml --fix
```

## Features

### Core

| Category | Features |
|----------|----------|
| **Scheduling** | YAML DAG format, dependency validation, cycle detection (DFS), parallel execution with `max_parallel`, matrix expansion, batch task mode |
| **Engines** | `codex exec`, `claude --print`, `gemini`, `copilot` (autopilot), `qwen`, `ollama` (local), `llama` (local via llama-cpp), raw shell commands; execution profiles (plan/safe/yolo) |
| **Prompts** | Inline, file, or markdown extraction; inter-task context passing (9 modes: raw/selective/summarized/map_reduce/recursive/layered/structural/council/knowledge_graph); progressive compaction; privacy pipeline (`output_redact`, `context_allowlist`); `append_system_prompt` |
| **Reliability** | `verify_command`, workspace assertions (`assert:`), `max_retries` with feedback injection, retry strategies (constant/linear/exponential), `allow_failure`, auto-escalation, cross-engine fallback, circuit breakers, `checkpoint` protocol |
| **Relational safety** | Typed `contract_type:` producers, `consumes_contracts:` consumers, `consistency_group:` membership, `reconcile_after:` group gates |
| **Cost control** | Per-task cost/token tracking, budget limits (`max_cost_usd`), cross-run budget tracking (`budget_period`), per-engine pricing tables, budget warning thresholds |
| **Quality gates** | LLM-as-Judge with typed assertions, Likert rubrics, G-Eval, adversarial debate judge (`method: debate`), comparative eval, named presets, `guard_command`, quorum voting, timeout auto-scaling |
| **Adaptive** | Mid-task signals (`signals: true`) — progress, metrics, timeout extension, budget query, artifacts; dynamic task decomposition (`dynamic_group: true`); cross-run knowledge auto-injection with prompt-relevant retrieval + `{{ knowledge_index }}`; adaptive temporal routing with trend detection + cross-task affinity; population-based search (best-of-N models); MCTS workflow search (draft/debug/improve, UCB1, `tree.jsonl`); self-evolving `replan` (multi-variant, tournament, elitism, diversity, novelty/knowledge priors, stepping stones) |
| **Knowledge + memory** | SQLite-backed per-plan memory store (`.maestro-cache/memory/<plan>.db`) with WAL, automatic JSONL migration, time-decayed confidence, bi-temporal records (`valid_from`/`valid_to` + `recorded_at`), provenance/trust labels, conflict resolution, relation confidence, poisoning quarantine, and score history (`ScoreRecord`, `plan_hash`, `quality_score`) |
| **Plan intelligence** | `deliberation: true` — haiku pre-call skips engine if task is self-answerable; `maestro validate` prints DAG density report (S_complex, W17/W18/W19 warnings); `output_schema` for structured inter-task typed outputs |
| **Flow control** | Conditional execution (`when`), `fail_fast`, `--only`/`--skip` filtering, `--tags`/`--skip-tags`, approval gates |
| **Secrets** | `secrets:` plan field (explicit list or `auto`), `--mask-secrets` CLI flag |
| **Imports** | `imports:` for reusable task templates with prefix namespacing, nested imports, cycle detection |
| **Policies** | Declarative runtime policies (`block`/`warn`/`audit`) with safe AST evaluation |
| **Caching** | Policy-versioned SHA-256 Merkle DAG keys, short-lived negative cache (`negative_cache_ttl_sec`), cache bypass for untrusted/tainted/partial/tool-failure results, pre-hash normalization (model alias resolution, sorted args), eviction "why" fields (`_cache_why`), `--no-cache`, `maestro explain` / `maestro status` |
| **Event sourcing** | Hash-chained `events.jsonl` with tamper detection; `maestro verify` validates integrity |
| **Security** | `maestro audit` scans plans (SEC001-SEC023), `--fix` auto-remediation, `--coverage` per-category breakdown; `allowed_tools:` per-task tool restriction; untrusted context detection and taint propagation; control flow integrity; trajectory guardrails (`trajectory_guard`); semantic firewall MCP metadata filtering with role and concurrency-safety hints; phantom workspace for destructive commands; git worktree isolation per task |
| **Blame** | `maestro blame` traces failure causality, classifies root causes, suggests fixes |
| **Watch** | Autonomous iteration loops (`maestro watch`): custom metric-driven mode and built-in plan improvement mode (`mode: improve`); git commit/rollback, consolidation agent with safety gates (trust labels, instructionality rejection, firewall), experiments.jsonl |

### Output Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Text** | `--output text` | Default. Human-readable `[maestro]` console output |
| **JSONL** | `--output jsonl` | Structured JSON Lines events to stdout; suppresses text |
| **Live** | `--output live` | Real-time Rich table with task progress, cost, duration. `pip install maestro-cli[live]` |
| **TUI** | `--output tui` | Interactive Textual app with DAG panel, detail panel, event feed, keyboard nav. `pip install maestro-cli[tui]` |

### Observability

| Feature | Description |
|---------|-------------|
| **Token tracking** | `TokenUsage` per task (input/cached/output), aggregated in manifest and summary |
| **Web UI** | Dashboard with stats, charts, cost trends, and collaboration summaries; run detail with ownership/blockers/activity, Gantt timeline, and log viewer |
| **Structured errors** | Error codes E001-E072 (validation), E100-E110 (runtime); warning codes W1-W30 |
| **Mid-task signals** | `task_progress`, `task_metric`, `task_artifact`, `timeout_extended` events from running tasks |
| **OTel export** | `maestro export-otel` emits per-task spans with `gen_ai.*` attributes, optional content capture/redaction, and task events such as `knowledge_poison_alert` and `memory_write` |
| **Diagnostics** | `maestro doctor` checks Python, PyYAML, engine CLIs, plugins, Git; `--full` adds cache, knowledge, skills, plans, prior runs |

## Plan Schema (compact)

```yaml
version: 1
name: my-plan
workspace_root: /path/to/project
max_parallel: 3
max_cost_usd: 25.00
budget_warning_pct: 0.8

defaults:
  timeout_sec: 600
  claude:
    model: sonnet

tasks:
  - id: setup
    command: ["bash", "-c", "npm install"]
    tags: [infra]

  - id: implement
    depends_on: [setup]
    engine: claude
    model: haiku
    escalation: [haiku, sonnet, opus]
    negative_cache_ttl_sec: 300
    prompt: "Implement the feature..."
    verify_command: "npm test"
    max_retries: 2

  - id: review
    depends_on: [implement]
    context_from: [implement]
    context_mode: summarized
    engine: claude
    cache: false
    prompt: "Review changes: {{ implement.summary }}"
    judge:
      method: g_eval
      criteria:
        - type: rubric
          name: correctness
          levels:
            - { score: 1, description: "Major bugs" }
            - { score: 5, description: "Correct and well-tested" }
      pass_threshold: 0.7
      on_fail: warn

  - id: deploy
    depends_on: [review]
    when: "{{ review.status }} == success"
    requires_approval: true
    command: "echo 'Deploying...'"
```

Full annotated schema with all fields: [docs/PLAN_GUIDE.md](docs/PLAN_GUIDE.md)

## CLI Commands

| Group | Commands |
|-------|----------|
| **Plan lifecycle** | `validate`, `run`, `replan`, `scaffold`, `watch` |
| **Observability** | `report`, `diff`, `explain`, `status`, `eval`, `suggest`, `blame` |
| **Security** | `audit`, `verify` |
| **Infrastructure** | `doctor`, `ci`, `ci-analyze`, `cleanup`, `backfill-costs`, `ui`, `mcp-server`, `export-otel` |
| **Interactive** | `chat`, `shell` |
| **Discovery** | `skill` (`list`, `search`, `recommend`) |

Key `run` flags: `--dry-run`, `--max-parallel N`, `--execution-profile plan|safe|yolo`, `--only`/`--skip`, `--tags`/`--skip-tags`, `--resume-last`, `--output text|jsonl|live|tui`, `--mask-secrets`, `--auto-approve`, `--no-cache`

Full CLI reference with all flags: [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md)

## Writing Effective Plans

**Prompt sources**: inline `prompt:`, file via `prompt_file:`, or markdown extraction via `prompt_md_file:` + `prompt_md_heading:`.

**Context passing**: use `context_from:` to inject upstream outputs. Nine modes: `raw` (free), `selective` (BM25 chunk-level, free), `summarized` (haiku per upstream), `map_reduce` (N haiku + synthesis), `recursive` (full workspace awareness), `layered` (L0/L1/L2 budget-aware tiers), `structural` (package-aware symbol extraction with re-export resolution and PageRank-based blast-radius scoring, free), `council` (multi-model deliberation with star/chain/graph topologies), `knowledge_graph` (entity extraction, free). Control budget with `context_budget_tokens:`. Add `context_compaction: progressive` for staged degradation. Use `maestro explain --context` to see why context was selected.

**Cross-run memory**: full runs accumulate reusable lessons in `.maestro-cache/memory/<plan>.db`. Matching records are injected automatically as `{{ task_knowledge }}`, and a compact `{{ knowledge_index }}` is available when you want the model to choose from a wider memory surface without pasting every record.

**Semantic firewall**: set top-level `firewall_model: haiku` to enable an opt-in second pass for MCP metadata and tainted upstream text (`stdout_tail`, `result_text`, `summary`). Classifier failures fail open to the deterministic sanitizer rather than blocking task execution. For MCP providers that mutate shared local state, set `mcp_servers[].is_concurrency_safe: false` so parallel `worktree: true` tasks are serialized around that provider.

**Verify + retry**: add `verify_command:` + `max_retries:` -- failure output is auto-injected into the retry prompt. Add `retry_delay_sec: [2, 5, 15]` for backoff.

**Caching policy**: keep default positive caching for deterministic tasks. For flaky or rate-limited tasks, set `negative_cache_ttl_sec:` to suppress repeated failures for a short window without pinning bad results for too long. Results from untrusted/tainted/partial runs, plus structured tool-failure outputs, are excluded from positive cache writes.

**Tracing**: `maestro export-otel` converts completed runs into OTLP spans or JSON, attaching `gen_ai.*` attributes, optional task input/output previews via `--include-content`, and prompt redaction via `--otel-mask-prompts`. Span events include runtime signals such as `knowledge_poison_alert` and `memory_write`.

**Quality gates**: `judge:` block with typed assertions (zero-cost `contains`/`regex`/`is-json`) and LLM rubrics. `guard_command:` for lightweight stdin-pipe validation.

**Budget**: `max_cost_usd:` at plan level. Set `budget_warning_pct: 0.8` for early alerts. Track across runs with `budget_period: weekly`.

**Conditional execution**: `when: "{{ task.status }} == success"` enables deploy-on-success / rollback-on-failure patterns.

**Signals**: `signals: true` on long-running tasks for progress reporting, timeout extension, and budget queries via `[MAESTRO_SIGNAL]` stdout protocol.

**Watch loops**: `maestro watch plan.yaml` for autonomous metric-driven iteration. `mode: improve` auto-fixes plan failures.

Complete guide with examples: [docs/PLAN_GUIDE.md](docs/PLAN_GUIDE.md)

## Models (quick reference)

| Engine | Example Aliases | Reasoning Effort | Cost Model |
|--------|----------------|-----------------|------------|
| **Claude** | `haiku`, `sonnet`, `opus` | `low`/`medium`/`high` (Opus only) | Per-token |
| **Codex** | `5-mini`, `5.1`, `5.4` | `minimal`→`xhigh` | Per-token |
| **Gemini** | `flash`, `pro`, `pro-3.1` | N/A (use model selection) | Per-token |
| **Copilot** | `sonnet`, `gpt-5.4-codex`, `grok` | N/A | Subscription |
| **Qwen** | `coder`, `max`, `qwq` | N/A | Per-token |
| **Ollama** | `llama3`, `codellama`, `mixtral` | N/A | Free (local) |
| **Llama** | `llama3`, `codellama`, `mistral` | N/A | Free (local, llama-cpp) |

Set `model: auto` for automatic routing based on task complexity. Control bias with `routing_strategy: cost_optimized | quality_first | balanced`.

Full model alias tables and pricing: [docs/MODELS.md](docs/MODELS.md)

## Execution Profiles

| Profile | Codex | Claude | Gemini | Copilot | Qwen | Ollama | Llama |
|---------|-------|--------|--------|---------|------|--------|-------|
| `plan` | Use YAML args exactly | Use YAML args exactly | Use YAML args exactly | Use YAML args exactly | Use YAML args exactly | Use YAML args exactly | Use YAML args exactly |
| `safe` | Forces sandbox + approval gates | Forces default permissions | Strips dangerous flags | Strips `--yolo` | Strips `--yolo` | No change | No change |
| `yolo` | Ensures full bypass | Ensures `--dangerously-skip-permissions` | Ensures `--approval-mode yolo` | Ensures `--yolo` | Ensures `--yolo` | No change | No change |

## Output Structure

```
.maestro-runs/<timestamp>_<plan-name>/
├── .cache/               # Task result cache (content-addressable)
├── run_manifest.json      # Aggregated results (status, cost, tokens, plan_hash/quality_score on full runs)
├── run_summary.md         # Human-readable summary with cost/timing/tokens
├── events.jsonl           # Structured event log (hash-chained, tamper-detectable)
├── <task-id>.log          # Execution transcript
└── <task-id>.result.json  # Structured result (status, exit_code, duration, cost, tokens)
```

Cross-run state lives separately in `.maestro-cache/`: task cache entries, knowledge JSONL from older versions, and the current SQLite memory store under `.maestro-cache/memory/`.

## Architecture

```
src/maestro_cli/
├── cli.py             # argparse CLI (27 subcommands) + --version + banner
├── models.py          # Dataclasses and typed results (PlanSpec, TaskSpec, TaskResult, ScoreRecord, ...)
├── loader.py          # YAML parsing + validation + cycle detection + matrix + imports
├── runners.py         # Command building + execution + verify/retry + judge + signals + secrets
├── scheduler.py       # DAG scheduler (ThreadPoolExecutor) + context + policies + budget getter
├── routing.py         # Semantic model routing (model: auto) + predictive + temporal + cross-task
├── memory.py          # SQLite-backed Knowledge + Memory v2 (WAL, provenance, score history)
├── policy.py          # Declarative runtime policy engine (safe AST)
├── blame.py           # Causal failure attribution
├── audit.py           # Plan security scanner (SEC001-SEC023)
├── eventsource.py     # Hash-chained event sourcing + verify
├── replan.py          # Adaptive re-planning + multi-variant search + knowledge-guided scoring
├── mcts.py            # MCTS workflow search (WorkflowVariant tree, selection, simulation, pruning)
├── watch.py           # Autonomous metric-driven iteration loop (custom + improve mode)
├── worktree.py        # Git worktree isolation per task
├── dynamic.py         # Dynamic task decomposition (dynamic_group runtime sub-plans)
├── knowledge.py       # Cross-run knowledge accumulation + consolidation + auto-inject
├── contracts.py       # Typed contract normalization (sql-schema, api-schema, etc.)
├── budget.py          # Cross-run budget tracking and ledger
├── chat.py            # Multi-model interactive terminal
├── shell.py           # Interactive REPL with slash commands
├── plugins.py         # Custom engine plugin discovery
├── multi.py           # Multi-plan execution
├── live.py            # Rich live output + signal progress (--output live)
├── tui/               # Textual TUI + signal events (--output tui)
├── web/               # FastAPI + vanilla JS dashboard
├── cache.py           # Task/plan hashing + semantic cache policy
├── codebase_graph.py  # AST-backed codebase graph + blast radius analysis
├── scaffold.py        # Plan generation from briefs + workflow libraries
├── skill_registry.py  # Skill discovery, search, and recommendation (maestro skill)
├── ci_agent.py        # CI failure analysis and remediation (maestro ci-analyze)
├── suggest.py         # Run history analysis + optimization heuristics
├── eval.py            # Batch judge evaluation
├── diff.py            # Run comparison
├── explain.py         # Cache hit/miss explanation + context trajectory
├── status.py          # Task staleness detection
├── report.py          # HTML report generation
├── cost_backfill.py   # Historical cost/token backfill
├── cleanup.py         # Run directory cleanup
├── doctor.py          # Environment diagnostics
├── ag_ui.py           # AG-UI protocol adapter (event translation + state tracking)
├── mcp_server.py      # MCP server (12 tools, 8 resources, 3 prompts)
├── otel.py            # OTLP exporter (run → OpenTelemetry spans)
├── council.py         # Multi-model deliberation (star/chain/graph topologies)
├── knowledge_graph.py # Entity extraction + graph-based context
├── symbols.py         # Regex-based code symbol extraction (10 languages)
├── errors.py          # PlanValidationError (E001-E072), TaskExecutionError (E100-E110)
└── utils.py           # Paths, templates, markdown extraction
```

**Design principles**: local-first persistence (run artifacts + SQLite memory), engine-agnostic (shells out to CLIs), minimal deps (only PyYAML in core; SQLite is stdlib), strong typing (dataclasses throughout).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[E0xx] message` | Plan validation error | Check error code, fix the YAML |
| `Heading not found in markdown` | `prompt_md_heading` includes `## ` | Remove the `## ` prefix |
| `No code fence found` | Prompt text not in `` ```text `` block | Wrap prompt in code fence |
| Task immediately fails | Engine CLI not on PATH | Run `maestro doctor` |
| Copilot auth fails | GitHub token missing/expired | Set `COPILOT_GITHUB_TOKEN` or `GH_TOKEN` |
| `exit_code: 124` | Task timed out | Increase `timeout_sec` |
| Task skipped | Dependency failed or `fail_fast` | Fix upstream, use `--resume-last` |
| Shell fails on Windows | `shell: true` uses `cmd.exe` | Use list-format command with Git Bash |
| Budget exceeded | `max_cost_usd` limit reached | Increase limit or optimize model selection |
| Judge times out | `g_eval` + many criteria | Set `judge.timeout_sec: 180` or let auto-scaling handle it |
| Worktree merge conflict | Parallel tasks edit same files | Use `worktree: true` with non-overlapping tasks, or add a reconciler |

For Windows-specific pitfalls, see [docs/PITFALLS.md](docs/PITFALLS.md).

---

## Roadmap

See [CHANGELOG.md](CHANGELOG.md) for full release history and [docs/ROADMAP.md](docs/ROADMAP.md) for planned features.

**Current repo state**: `v2.4.0` is the latest tagged release; `main` also carries early `v2.5.0` session-memory foundations for long-horizon `watch` loops on top of the completed Phase 3 work. 7 engines, 9 context modes, SQLite-backed memory, durable watch `session_snapshots`, and ~11.3K tests in the latest full-suite run.

## Requirements

- Python >= 3.11
- PyYAML >= 6.0
- Engine CLIs on PATH: `codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`, `llama` (as needed)
- Optional: `[live]` (Rich), `[tui]` (Textual), `[web]` (FastAPI + uvicorn), `[agui]` (AG-UI protocol), `[mcp]` (MCP SDK), `[otel]` (OpenTelemetry OTLP exporter)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, CI, and code conventions.

Full documentation index: [docs/README.md](docs/README.md).

## License

Maestro CLI is released under the [MIT License](LICENSE).

Copyright (c) 2026 Tiago Perez
