```
  ██   ██  ████  ██████  ████ ██████ █████   ████
  ███ ███ ██  ██ ██     ██      ██   ██  ██ ██  ██
  ██ █ ██ ██████ █████   ███    ██   █████  ██  ██
  ██   ██ ██  ██ ██        ██   ██   ██ ██  ██  ██
  ██   ██ ██  ██ ██████ ████    ██   ██  ██  ████

  Maestro CLI -- Version 2.5.1
  CLI orchestrator for multi-step AI execution plans
```

# Maestro CLI

[![CI](https://github.com/tiagojcperez/maestro-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/tiagojcperez/maestro-cli/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tiagojcperez/maestro-cli/branch/main/graph/badge.svg)](https://codecov.io/gh/tiagojcperez/maestro-cli)
[![Coverage Status](https://coveralls.io/repos/github/tiagojcperez/maestro-cli/badge.svg?branch=main)](https://coveralls.io/github/tiagojcperez/maestro-cli?branch=main)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=tiagojcperez_maestro-cli&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=tiagojcperez_maestro-cli)
[![PyPI](https://img.shields.io/pypi/v/maestro-ai-cli.svg)](https://pypi.org/project/maestro-ai-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Maestro turns a YAML file into a parallel, dependency-aware pipeline of AI agents and shell commands.**
Instead of gluing engine CLIs together with bash, you declare the pipeline once and get parallel DAG
scheduling, context passing between steps, cost budgets, quality gates, and deterministic, replayable logs
for free -- across Claude, Codex, Gemini, Copilot, Qwen, Ollama, and Llama, from one dependency-light CLI
(PyYAML is the only required dependency; everything else is stdlib or optional).

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

> That example needs the `claude` CLI on PATH. To try Maestro with **zero setup and no API keys**, run the engine-free [`examples/demo_plan.yaml`](examples/demo_plan.yaml) from the [Quickstart](#quickstart) below.

![Maestro CLI demo](docs/assets/demo.gif)

### Supported Engines

<p>
<img src="https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=claude&logoColor=white" alt="Claude"> <img src="https://img.shields.io/badge/Codex-412991?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0yMi4yODIgOS44MjFhNS45ODUgNS45ODUgMCAwIDAtLjUxNi00LjkxIDYuMDQ2IDYuMDQ2IDAgMCAwLTYuNTEtMi45QTYuMDY1IDYuMDY1IDAgMCAwIDQuOTgxIDQuMThhNS45ODUgNS45ODUgMCAwIDAtMy45OTggMi45IDYuMDQ2IDYuMDQ2IDAgMCAwIC43NDMgNy4wOTcgNS45OCA1Ljk4IDAgMCAwIC41MSA0LjkxMSA2LjA1MSA2LjA1MSAwIDAgMCA2LjUxNSAyLjlBNS45ODUgNS45ODUgMCAwIDAgMTMuMjYgMjRhNi4wNTYgNi4wNTYgMCAwIDAgNS43NzItNC4yMDYgNS45OSA1Ljk5IDAgMCAwIDMuOTk3LTIuOSA2LjA1NiA2LjA1NiAwIDAgMC0uNzQ3LTcuMDczek0xMy4yNiAyMi40M2E0LjQ3NiA0LjQ3NiAwIDAgMS0yLjg3Ni0xLjA0bC4xNDEtLjA4MSA0Ljc3OS0yLjc1OGEuNzk1Ljc5NSAwIDAgMCAuMzkyLS42ODF2LTYuNzM3bDIuMDIgMS4xNjhhLjA3MS4wNzEgMCAwIDEgLjAzOC4wNTJ2NS41ODNhNC41MDQgNC41MDQgMCAwIDEtNC40OTQgNC40OTR6TTMuNiAxOC4zMDRhNC40NyA0LjQ3IDAgMCAxLS41MzUtMy4wMTRsLjE0Mi4wODUgNC43ODMgMi43NTlhLjc3MS43NzEgMCAwIDAgLjc4IDBsNS44NDMtMy4zNjl2Mi4zMzJhLjA4LjA4IDAgMCAxLS4wMzMuMDYyTDkuNzQgMTkuOTVhNC41IDQuNSAwIDAgMS02LjE0LTEuNjQ2ek0yLjM0IDcuODk2YTQuNDg1IDQuNDg1IDAgMCAxIDIuMzY2LTEuOTczVjExLjZhLjc2Ni43NjYgMCAwIDAgLjM4OC42NzdsNS44MTUgMy4zNTUtMi4wMiAxLjE2OGEuMDc2LjA3NiAwIDAgMS0uMDcxIDBsLTQuODMtMi43ODZBNC41MDQgNC41MDQgMCAwIDEgMi4zNCA3Ljg3MnptMTYuNTk3IDMuODU1bC01LjgzMy0zLjM4N0wxNS4xMTkgNy4yYS4wNzYuMDc2IDAgMCAxIC4wNzEgMGw0LjgzIDIuNzkxYTQuNDk0IDQuNDk0IDAgMCAxLS42NzYgOC4xMDV2LTUuNjc4YS43OS43OSAwIDAgMC0uNDA3LS42Njd6bTIuMDEtMy4wMjNsLS4xNDEtLjA4NS00Ljc3NC0yLjc4MmEuNzc2Ljc3NiAwIDAgMC0uNzg1IDBMOS40MDkgOS4yM1Y2Ljg5N2EuMDY2LjA2NiAwIDAgMSAuMDI4LS4wNjFsNC44My0yLjc4N2E0LjUgNC41IDAgMCAxIDYuNjggNC42NnptLTEyLjY0IDQuMTM1bC0yLjAyLTEuMTY0YS4wOC4wOCAwIDAgMS0uMDM4LS4wNTdWNi4wNzVhNC41IDQuNSAwIDAgMSA3LjM3NS0zLjQ1M2wtLjE0Mi4wOEw4LjcwNCA1LjQ2YS43OTUuNzk1IDAgMCAwLS4zOTMuNjgxem0xLjA5Ny0yLjM2NWwyLjYwMi0xLjUgMi42MDcgMS41djIuOTk5bC0yLjU5NyAxLjUtMi42MDctMS41eiIvPjwvc3ZnPg==&logoColor=white" alt="Codex"> <img src="https://img.shields.io/badge/Gemini-886FBF?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini"> <img src="https://img.shields.io/badge/Copilot-000?style=for-the-badge&logo=githubcopilot&logoColor=white" alt="Copilot"> <img src="https://img.shields.io/badge/Qwen-5A29E4?style=for-the-badge&logo=alibabadotcom&logoColor=white" alt="Qwen"> <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"> <img src="https://img.shields.io/badge/Llama-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="Llama">
</p>

**Contents:** [Why Maestro?](#why-maestro) · [Python SDK](#python-sdk) · [Install](#install) · [Quickstart](#quickstart) · [Features](#features) · [Plan Schema](#plan-schema-compact) · [CLI Commands](#cli-commands) · [Writing Plans](#writing-effective-plans) · [Models](#models-quick-reference) · [Architecture](#architecture) · [Troubleshooting](#troubleshooting)

## Why Maestro?

- **One YAML, many engines** -- orchestrate Claude, Codex, Gemini, Copilot, Qwen, Ollama, and Llama (plus raw shell commands) in a single plan, with per-engine model aliases and reasoning-effort control.
- **Parallel DAG scheduling** -- declare dependencies and Maestro runs tasks in the right order with configurable parallelism, matrix expansion, and nested sub-plans.
- **Context that flows** -- pass outputs between tasks with 10 context modes, from zero-cost BM25 selection to multi-model council deliberation, with token budgets and progressive compaction.
- **Quality gates built in** -- LLM-as-Judge (rubrics, G-Eval, debate, quorum), zero-cost typed assertions, and `verify_command` retries with feedback injection keep results honest.
- **Cost-aware and resilient** -- per-task and cross-run budgets, retries with backoff, auto-escalation to stronger models, cross-engine fallback, and circuit breakers.
- **Deterministic and observable** -- every run is logged to JSON/JSONL with hash-chained, tamper-detectable events; watch it live in a TUI, a Web UI, or `maestro report`.
- **Secure by default** -- untrusted-context taint tracking, prompt-injection containment (`allowed_tools:`), and 23 audit rules via `maestro audit`. The optional Web UI/API confines plan and run paths to the project root and limits CORS to same-machine origins by default.
- **Dependency-light** -- PyYAML is the only required dependency; everything else is stdlib or an optional extra.

For the exhaustive capability list, see [Features](#features) below. &nbsp;|&nbsp; v1 stability contract: [docs/V1_API_FREEZE.md](docs/V1_API_FREEZE.md) &nbsp;|&nbsp; Migration from 0.x: [docs/MIGRATING_TO_V1.md](docs/MIGRATING_TO_V1.md)

## Python SDK

Maestro CLI ships with a programmatic API (29 stable exports) and a `py.typed` marker for static analysis:

```python
from maestro_cli import load_plan, run_plan

plan = load_plan("plan.yaml")
result = run_plan(plan)
```

## Install

```bash
pip install maestro-ai-cli                          # core CLI
pip install "maestro-ai-cli[live]"                  # + Rich live table (--output live)
pip install "maestro-ai-cli[tui]"                   # + Textual TUI (--output tui)
pip install "maestro-ai-cli[web]"                   # + Web UI (FastAPI + uvicorn)
pip install "maestro-ai-cli[agui]"                  # + AG-UI protocol endpoint
pip install "maestro-ai-cli[mcp]"                   # + MCP server for IDE integration
pip install "maestro-ai-cli[otel]"                  # + OpenTelemetry OTLP exporter
pip install "maestro-ai-cli[live,web,tui,agui,mcp,otel]"   # everything
```

The package installs as `maestro-ai-cli`; the command is `maestro` and the Python
import is `maestro_cli`. For development, install from source instead:

```bash
git clone https://github.com/tiagojcperez/maestro-cli.git
cd maestro-cli && pip install -e ".[live,web,tui,agui,mcp,otel]"
```

Requires Python >= 3.11 and at least one engine CLI on PATH (`codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`, `llama`).

## Quickstart

```bash
# Validate a plan (examples/demo_plan.yaml runs with no API keys)
maestro validate examples/demo_plan.yaml
# -> "Plan is valid: ... name: demo, tasks: 4"

# Run it -- the demo uses only shell tasks, so no API keys are needed
maestro run examples/demo_plan.yaml
# -> "... 4 ok / 0 failed / 0 skipped"  (artifacts written to .maestro-runs/)

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

# Adaptive re-planning on failure (multi-variant search, tournament selection,
# elitism and diversity floors are documented in docs/CLI_REFERENCE.md)
maestro replan examples/demo_plan.yaml --max-attempts 3

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

**`maestro audit` catches dangerous plans before they run** — a structurally *valid* plan can still be *unsafe* (unbounded spend, `rm -rf`, yolo bypass flags):

![Maestro audit](docs/assets/demo-audit.gif)

**Next steps:** write your own plan with the [Plan Guide](docs/PLAN_GUIDE.md) -> copy a ready-made recipe from the [Playbook](docs/PLAYBOOK.md) -> look up any flag in the [CLI Reference](docs/CLI_REFERENCE.md).

## Features

### Core

| Category | Features |
|----------|----------|
| **Scheduling** | YAML DAG format, dependency validation, cycle detection (DFS), parallel execution with `max_parallel`, matrix expansion, batch task mode |
| **Engines** | `codex exec`, `claude --print`, `gemini`, `copilot` (autopilot), `qwen`, `ollama` (local), `llama` (local via llama-cpp), raw shell commands; execution profiles (plan/safe/yolo) |
| **Prompts & context** | Inline, file, or markdown extraction; inter-task context passing (10 modes: raw/selective/summarized/map_reduce/recursive/layered/structural/council/knowledge_graph/codebase_map); progressive compaction; privacy pipeline (`output_redact`, `context_allowlist`) |
| **Reliability** | `verify_command`, workspace assertions (`assert:`), `max_retries` with feedback injection, retry strategies (constant/linear/exponential), `allow_failure`, auto-escalation, cross-engine fallback, circuit breakers, `checkpoint` protocol |
| **Cost control** | Per-task cost/token tracking, budget limits (`max_cost_usd`), cross-run budget tracking (`budget_period`), per-engine pricing tables, budget warning thresholds |
| **Quality gates** | LLM-as-Judge with typed assertions, Likert rubrics, G-Eval, adversarial debate judge, comparative eval, named presets, `guard_command`, quorum voting, timeout auto-scaling |
| **Security** | `maestro audit` (SEC001-SEC023) + `--fix`; `allowed_tools:` per-task restriction; untrusted-context detection and taint propagation; control flow integrity; trajectory guardrails; semantic firewall for MCP metadata; phantom workspace; git worktree isolation per task; Web UI/API path confinement + localhost-default CORS |
| **Caching** | Policy-versioned SHA-256 Merkle DAG keys, short-lived negative cache (`negative_cache_ttl_sec`), contamination-aware bypass, pre-hash normalization, eviction "why" fields, `--no-cache`, `maestro explain` / `maestro status` |
| **Flow control** | Conditional execution (`when`), `fail_fast`, `--only`/`--skip` filtering, `--tags`/`--skip-tags`, approval gates |
| **Secrets** | `secrets:` plan field (explicit list or `auto`), `--mask-secrets` CLI flag |

<details>
<summary><b>Advanced capabilities</b> — relational contracts, adaptive search, persistent memory, policies, watch loops, blame, event sourcing</summary>

| Category | Features |
|----------|----------|
| **Relational safety** | Typed `contract_type:` producers, `consumes_contracts:` consumers, `consistency_group:` membership, `reconcile_after:` group gates |
| **Adaptive** | Mid-task signals (`signals: true`) — progress, metrics, timeout extension, budget query, artifacts; dynamic task decomposition (`dynamic_group: true`); cross-run knowledge auto-injection with prompt-relevant retrieval + `{{ knowledge_index }}`; adaptive temporal routing with trend detection + cross-task affinity; population-based search (best-of-N models); MCTS workflow search (draft/debug/improve, UCB1, `tree.jsonl`); self-evolving `replan` (multi-variant, tournament, elitism, diversity, novelty/knowledge priors, stepping stones) |
| **Knowledge + memory** | SQLite-backed per-plan memory store (`.maestro-cache/memory/<plan>.db`) with WAL, automatic JSONL migration, time-decayed confidence, bi-temporal records (`valid_from`/`valid_to` + `recorded_at`), provenance/trust labels, conflict resolution, relation confidence, poisoning quarantine, and score history (`ScoreRecord`, `plan_hash`, `quality_score`) |
| **Plan intelligence** | `deliberation: true` — haiku pre-call skips engine if task is self-answerable; `maestro validate` prints DAG density report (S_complex, W17/W18/W19 warnings); `output_schema` for structured inter-task typed outputs |
| **Imports** | `imports:` for reusable task templates with prefix namespacing, nested imports, cycle detection |
| **Policies** | Declarative runtime policies (`block`/`warn`/`audit`) with safe AST evaluation |
| **Event sourcing** | Hash-chained `events.jsonl` with tamper detection; `maestro verify` validates integrity |
| **Blame** | `maestro blame` traces failure causality, classifies root causes, suggests fixes |
| **Watch** | Autonomous iteration loops (`maestro watch`): custom metric-driven mode and built-in plan improvement mode (`mode: improve`); git commit/rollback, consolidation agent with safety gates (trust labels, instructionality rejection, firewall), experiments.jsonl |

</details>

### Output Modes

| Mode | Flag | Description |
|------|------|-------------|
| **Text** | `--output text` | Default. Human-readable `[maestro]` console output |
| **JSONL** | `--output jsonl` | Structured JSON Lines events to stdout; suppresses text |
| **Live** | `--output live` | Real-time Rich table with task progress, cost, duration. `pip install maestro-ai-cli[live]` |
| **TUI** | `--output tui` | Interactive Textual app with DAG panel, detail panel, event feed, keyboard nav. `pip install maestro-ai-cli[tui]` |

![Maestro TUI](docs/assets/demo-tui.gif)

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
| **Plan lifecycle** | `validate`, `check`, `run`, `replan`, `scaffold`, `watch` |
| **Observability** | `report`, `diff`, `explain`, `status`, `eval`, `suggest`, `blame`, `budget` |
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
| **Claude** | `haiku`, `sonnet`, `opus` | `low` to `max` (Opus; `xhigh`/`max` are Opus-tier) | Per-token |
| **Codex** | `5.4-mini`, `5.4`, `5.5` | `none` to `xhigh` | Per-token |
| **Gemini** | `flash`, `pro`, `pro-3.1` | N/A (use model selection) | Per-token |
| **Copilot** | `sonnet`, `gpt-5.4-codex`, `gemini-pro` | N/A | Subscription |
| **Qwen** | `coder`, `max`, `qwq` | N/A | Per-token |
| **Ollama** | `llama4`, `qwen3-coder`, `deepseek-r1` | N/A | Free (local) |
| **Llama** | `llama3`, `llama4-scout`, `codellama` | N/A | Free (local, llama-cpp) |

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

## Testing & Guarantees

Maestro is tested offline-first. The full suite (13k+ tests) runs on every push
across Python 3.11 / 3.12 / 3.13 plus a Windows lane, alongside strict `mypy`,
a documentation lint, and CodeQL. Engine calls are **mocked** in CI; the
real-engine end-to-end tests are opt-in (`MAESTRO_RUN_REAL_ENGINE_TESTS=1`, they
need provider credentials and cost money) and run on a separate, manually-enabled
lane. Coverage is uploaded to Codecov and Coveralls (with optional SonarCloud and
Codacy backends — see [docs/COVERAGE_PLATFORMS.md](docs/COVERAGE_PLATFORMS.md)).

| Area | Unit | Integration | Real-engine |
|------|:----:|:-----------:|:-----------:|
| DAG scheduling / dependencies | ✅ | ✅ | n/a |
| Shell tasks | ✅ | ✅ | n/a |
| Engines (codex/claude/gemini/copilot/qwen/ollama/llama) | ✅ (mocked) | ✅ (command build) | opt-in |
| Context passing (10 modes) | ✅ | ✅ | partial |
| Budgets / cost / token tracking | ✅ | ✅ | partial |
| Quality gates (judge / verify / guard / assert) | ✅ | ✅ | partial |
| Retries / fallback / circuit breakers | ✅ | ✅ | n/a |
| Secret masking | ✅ | ✅ | n/a |
| Policy engine (safe AST) | ✅ (fuzzed) | ✅ | n/a |
| Security audit (SEC001-SEC023) | ✅ | ✅ | n/a |
| Cache / event sourcing / blame | ✅ | ✅ | n/a |

**Maestro guarantees** DAG ordering and dependency semantics, retry/fallback
behaviour, the documented `version: 1` plan schema and run-artifact shapes (the
[v1 stability contract](docs/V1_API_FREEZE.md)), best-effort secret masking, and
deterministic, replayable run logs.

**Maestro does not guarantee** deterministic LLM output, provider/CLI availability,
model-pricing accuracy (the pricing tables are best-effort and overridable), or the
safety of arbitrary user-authored shell commands — you own the plans you run.

For the full per-engine breakdown (which engines have real end-to-end tests, what
runs in default CI, and the outstanding gaps stated honestly), see
[docs/TESTED_GUARANTEES.md](docs/TESTED_GUARANTEES.md).

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

**Current repo state**: `v2.5.1` is the latest release — adds the `codebase_map` context mode (consume an Understand-Anything knowledge graph) plus a quality/CI patch tranche (SonarCloud all-A, scan-action v6, 117 dead tests un-shadowed) on top of the v2.5.0 security hardening. 7 engines, 10 context modes, SQLite-backed memory, durable watch `session_snapshots`, and ~13.4K tests in the latest full-suite run.

## Requirements

- Python >= 3.11
- PyYAML >= 6.0
- Engine CLIs on PATH: `codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`, `llama` (as needed)
- Optional: `[live]` (Rich), `[tui]` (Textual), `[web]` (FastAPI + uvicorn), `[agui]` (AG-UI protocol), `[mcp]` (MCP SDK), `[otel]` (OpenTelemetry OTLP exporter)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, CI, and code conventions. Please also read our [Code of Conduct](CODE_OF_CONDUCT.md), and the [security policy](SECURITY.md) for reporting vulnerabilities privately.

Full documentation index: [docs/README.md](docs/README.md).

## License

Maestro CLI is released under the [MIT License](LICENSE).

Copyright (c) 2026 Tiago Perez
