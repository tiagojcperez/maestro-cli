# CLI Reference

Complete reference for all `maestro` CLI commands and flags.

> Back to [README](../README.md)

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tasks succeeded (or soft_failed with `allow_failure`) |
| 1 | Plan validation error or task failure |
| 2 | CLI argument error |

---

## Plan Lifecycle

### `maestro validate <plan.yaml>`

Validates plan structure, dependency graph (cycle detection), and field types. Prints non-blocking warnings.

### `maestro check <plan.yaml> [options]`

Run `validate` + `audit` in one pass with a single exit code. This is the recommended first-run / CI entrypoint: authors run validate and audit together nearly all of the time, and `check` combines both (plus optional suggestions) into one command.

| Flag | Description |
|------|-------------|
| `--json` | Emit a structured JSON report instead of formatted text |
| `--with-suggest` | Also include `maestro suggest` output (requires prior runs) |
| `--strict` | Exit non-zero on any warning (validation OR audit), not just errors |
| `--run-dir DIR` | Override run directory for `--with-suggest` (default: plan's `run_dir`) |

### `maestro run <plan.yaml> [more-plans...] [options]`

| Flag | Description |
|------|-------------|
| `--dry-run` | Build commands and schedule, but don't execute |
| `--max-parallel N` | Override plan's `max_parallel` |
| `--execution-profile plan\|safe\|yolo` | Safety mode (aliases: `--profile-mode`, `--mode`) |
| `--only t1,t2` | Run only these task IDs + their dependencies (repeatable) |
| `--skip t1,t2` | Exclude these task IDs (repeatable) |
| `--tags t1,t2` | Run only tasks with these tags (comma-separated) |
| `--skip-tags t1,t2` | Skip tasks with these tags (comma-separated) |
| `--mask-secrets` | Auto-detect and mask secret values in logs |
| `--auto-approve` | Auto-approve all approval gates (for pre-authorized CI runs) |
| `--run-dir DIR` | Override output directory |
| `--webhook URL` | Override plan `webhook_url` for run-complete notifications |
| `--output text\|jsonl\|live\|tui` | Output format: text (default), JSONL events, Rich live table, or Textual TUI |
| `--resume PATH` | Resume from a prior run directory (skip succeeded tasks) |
| `--resume-last` | Resume from the most recent run of this plan |
| `--no-cache` | Disable content-addressable caching (re-run all tasks) |
| `--cache-dir DIR` | Cache directory for task results (default: `<run-dir>/.cache`) |
| `--verbose, -v` | Show more output: task logs on success, extra detail |
| `--quiet, -q` | Suppress informational output; show only errors and final summary |
| `--parallel` | Run multiple plans in parallel (default: sequential) |
| `--set KEY=VALUE` | Set template variable (repeatable, e.g. `--set env=prod --set region=eu`) |

### `maestro replan <plan.yaml> [options]`

Adaptive re-planning: run the plan, analyze failures with a frontier model, generate corrected plans, and re-run. Includes diff-based approval, circuit breaker, and optional multi-variant search.

| Flag | Description |
|------|-------------|
| `--max-attempts N` | Maximum re-plan attempts (default: 3) |
| `--model MODEL` | Model for failure analysis (default: `opus`) |
| `--variants N` | Generate N corrected-plan candidates per failed round (default: 1) |
| `--selection-policy debug_prob\|ucb1` | Leaf-selection policy for multi-variant search (default: `debug_prob`) |
| `--debug-prob P` | In multi-variant mode, probability of expanding an invalid leaf instead of the best valid leaf (default: 0.5) |
| `--exploration-constant C` | UCB1 exploration constant when `--selection-policy ucb1` (default: 1.41421356237) |
| `--population-strategy best\|tournament` | How to pick the next variant from the candidate population (default: `best`) |
| `--tournament-size N` | Contestant count when `--population-strategy tournament` (default: 2) |
| `--elite-count N` | Top-leaf count preserved in the selection pool across rounds (default: 1) |
| `--diversity-floor F` | Minimum mutation-signature distance enforced between tournament contestants before fallback sampling (default: 0.25; `0` disables) |
| `--auto-approve` | Skip diff approval prompts |
| `--dry-run` | Build commands without executing |
| `--execution-profile plan\|safe\|yolo` | Safety mode |
| `--verbose, -v` | Show more output |
| `--quiet, -q` | Suppress informational output |
| `--output text\|jsonl\|live\|tui` | Output format |
| `--set KEY=VALUE` | Set template variable (repeatable) |

### `maestro scaffold <brief.yaml> [options]`

Generate a complete plan from a simplified brief YAML with automatic model routing, quality gates, and anti-stalling prompts.

| Flag | Description |
|------|-------------|
| `-o, --output PATH` | Write plan to file (default: stdout) |
| `--validate` | Validate the generated plan |
| `--cost-check` | Run cost safety checks |
| `--library NAME\|PATH` | Use a built-in workflow library or external library YAML |
| `--list-libraries` | List built-in workflow libraries and exit |
| `--strict-defaults` | Inject sane first-run defaults (`timeout_sec=1500`, `retry_delay_sec=[60, 120]`, `max_cost_usd=10.0`, `budget_warning_pct=0.8`) to skip warning whack-a-mole |

### `maestro watch <plan.yaml> [options]`

Autonomous metric-driven iteration loop. Runs a plan repeatedly, extracts a numeric metric, keeps (git commit) or rolls back changes based on improvement.

| Flag | Description |
|------|-------------|
| `--dry-run` | Build commands without executing |
| `--execution-profile plan\|safe\|yolo` | Safety mode |
| `--max-parallel N` | Override plan's `max_parallel` |
| `--auto-approve` | Auto-approve all approval gates |
| `--verbose, -v` | Show more output |
| `--quiet, -q` | Suppress informational output |
| `--output text\|jsonl\|live\|tui` | Output format |
| `--mask-secrets` | Auto-detect and mask secret values in logs |
| `--resume-last` | Resume from previous `experiments.jsonl` |
| `--set KEY=VALUE` | Set template variable (repeatable) |

---

## Observability

### `maestro doctor [options]`

Diagnose environment health: Python version, PyYAML, engine CLIs on PATH, custom plugin discovery, Git, run directory access.

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |
| `--run-dir DIR` | Run directory to check (default: `.maestro-runs`) |
| `--full` | Run extended checks for cache, knowledge, skills, plans, and prior runs |
| `--hardware` | Report local hardware (GPU/VRAM via `nvidia-smi`), installed Ollama models, and llama.cpp `*.gguf` files, then exit. Powers hardware-aware `model: auto` routing for local engines. |

`maestro doctor` always reports custom engine plugin discovery from the `maestro_cli.engines`
entry-point group, even when no plugins are installed.

Example JSON detail:

```json
{"check":"engine_plugins","detail":"no custom engine plugins discovered in 'maestro_cli.engines'","status":"ok"}
{"check":"engine_plugins","detail":"discovered 1 custom engine plugin(s) in 'maestro_cli.engines': acme","status":"ok"}
```

### `maestro mcp-server`

Launch the MCP (Model Context Protocol) server for IDE integration.

Requires the `[mcp]` optional extra: `pip install maestro-ai-cli[mcp]`.

Exposes Maestro's subcommands as MCP tools, run artefacts as resources,
and plan templates as prompts. Any MCP-compatible client (Claude Code,
VS Code, Cursor) can consume these via stdio transport.

```bash
# Add to Claude Code
claude mcp add --transport stdio maestro -- maestro mcp-server
```

### `maestro export-otel <run-path> [options]`

Export a completed run as OpenTelemetry spans. Each task becomes a span
with engine, model, cost, tokens, status attributes, and `gen_ai.*`
semantic-convention fields when available. Exported task events include
signals such as `knowledge_poison_alert` and `memory_write`.

Requires the `[otel]` optional extra: `pip install maestro-ai-cli[otel]`.
Without the SDK, falls back to JSON output.

| Flag | Description |
|------|-------------|
| `--endpoint URL` | OTLP endpoint (default: JSON to stdout) |
| `--json` | Force JSON output even if SDK is installed |
| `--include-content` | Attach task input/output previews to exported spans |
| `--otel-mask-prompts` | Redact captured input/output content in exported spans |

`--otel-mask-prompts` only affects captured content fields. Model, token,
cost, and status metadata are still exported.

### `maestro report <run-path> [options]`

Generate a self-contained HTML report from a completed run directory.

| Flag | Description |
|------|-------------|
| `-o, --output PATH` | Output HTML path (default: `<run-path>/report.html`) |

### `maestro diff <run_a> <run_b> [options]`

Compare two run directories side-by-side: task status changes, cost/token/duration deltas, regressions and fixes.

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |

### `maestro explain <plan.yaml> [options]`

Show cache hit/miss status for each task in a plan. With `--context`, shows the full context retrieval trajectory — why each upstream was selected, its BM25 score, hop decay factor, and whether it was trimmed by the token budget.

| Flag | Description |
|------|-------------|
| `--cache-dir DIR` | Cache directory override (default: auto-detected from plan, usually `<run-dir>/.cache`) |
| `--context` | Show context retrieval trajectory: BM25 scores, hop decay factors, and budget trim decisions per upstream per task |
| `--json` | Machine-readable JSON output |

### `maestro status <plan.yaml> [options]`

Show task staleness versus the latest run for a plan.

| Flag | Description |
|------|-------------|
| `--cache-dir DIR` | Cache directory override (default: auto-detected from plan, usually `<run-dir>/.cache`) |
| `--run-dir DIR` | Override run directory when locating the latest run |
| `--json` | Machine-readable JSON output |

### `maestro eval <eval.yaml> <run-path> [options]`

Run a reusable judge suite over an existing run results with CI-friendly exit codes.

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |
| `--verbose, -v` | Show detailed judge reasoning |

### `maestro suggest <plan.yaml> [options]`

Suggest deterministic plan optimizations from run history.

| Flag | Description |
|------|-------------|
| `--run-dir DIR` | Override run directory |
| `--min-runs N` | Minimum runs to analyse (default: `3`) |
| `--json` | Machine-readable JSON output |

### `maestro estimate <plan.yaml> [options]`

Estimate the cost of running a plan **before** running it — read-only, offline, deterministic. Uses the average actual cost from prior runs when available (`history`), otherwise a transparent token heuristic from prompt size + the per-model pricing tables (`heuristic`, a lower bound). Shell tasks are free; `ollama`/`llama` are zero-cost (local); `copilot` is subscription-based.

| Flag | Description |
|------|-------------|
| `--run-dir DIR` | Override run directory for prior-run cost history |
| `--set KEY=VALUE` | Inject a template variable (repeatable) for prompt rendering |
| `--json` | Machine-readable JSON output |

### `maestro blame <run-path> [options]`

Trace failure causality in a completed run via dependency graph backward walk. Classifies root causes (timeout, budget, context, cascade), provides confidence scores and suggested fixes.

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |

### `maestro verify <run-path> [options]`

Verify hash-chain integrity of a run's `events.jsonl`. Detects tampered, reordered, or missing events.

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |

### `maestro backfill-costs [options]`

Backfill cost and token data for historical runs by scanning task logs.

| Flag | Description |
|------|-------------|
| `--root PATH` | Project root for auto-discovery (default: current dir) |
| `--run-root PATH` | Explicit `.maestro-runs` directory (repeatable) |
| `--dry-run` | Report without writing |
| `--codex-pricing-file PATH` | JSON pricing table for Codex token-based estimates |

---

## Security & Audit

### `maestro audit <plan.yaml> [options]`

Scan plans for security risks (SEC001-SEC023). Can auto-fix common issues with `--fix`.

| Flag | Description |
|------|-------------|
| `--json` | Machine-readable JSON output |
| `--fix` | Auto-remediate fixable findings (creates `.yaml.bak` backup) |
| `--coverage` | Show per-category coverage breakdown for built-in SEC rules |

Auto-fix rules: SEC001 (adds `max_cost_usd: 10.0`), SEC003/SEC014 (adds `secrets: auto`).

---

## Infrastructure

### `maestro ci <plan.yaml> [options]`

Generate starter CI config from an existing plan.

| Flag | Description |
|------|-------------|
| `--provider` | `github_actions`, `gitlab_ci`, `github`, or `gitlab` |
| `--output PATH` | Write generated YAML to a file instead of stdout |
| `--workflow-name TEXT` | Workflow/pipeline display name |
| `--python-version TEXT` | Python version used in generated bootstrap/install steps |
| `--test-command TEXT` | Test command used in generated test lanes |

Generator behaviour:

- Validates the plan first via `load_plan()`
- Generates explicit `linux`, `windows`, and `macos` validate/test lanes
- Adds a manual opt-in real-engine lane (not mandatory)
  - GitHub Actions: `maestro_real_engine` behind `workflow_dispatch`
  - GitLab CI: `run_maestro_real_engine` behind `MAESTRO_RUN_REAL_ENGINE=1`
- Uses shell-neutral `python -m ...` and `maestro ...` commands

`maestro ci` is part of the frozen `1.x` command surface as of v1.31.0. The generated YAML is frozen at the structural level documented in [V1_API_FREEZE.md](V1_API_FREEZE.md): existing validate/test/real-engine jobs or stages remain stable, while additive jobs/stages are still allowed.

### `maestro cleanup <plan.yaml> [options]`

| Flag | Description |
|------|-------------|
| `--keep N` | Keep N most recent runs (default: 10) |
| `--older-than DAYS` | Only delete runs older than N days |
| `--dry-run` | List directories that would be deleted |

### `maestro ui [options]`

Launch the Web UI dashboard (requires `pip install maestro-ai-cli[web]`).

| Flag | Description |
|------|-------------|
| `--host HOST` | Bind address (default: `127.0.0.1`) |
| `--port PORT` | Port number (default: `8000`) |
| `--no-browser` | Don't auto-open browser |
| `--project-root PATH` | Additional project root for run discovery (repeatable) |

---

## Interactive

### `maestro chat [options]`

Multi-model interactive terminal. Chat with any configured engine from a single prompt. Supports `@engine` prefix routing (e.g., `@codex optimize this`), conversation history, streaming output, and slash commands (`/model`, `/models`, `/context`, `/save`, `/load`, `/clear`, `/cost`, `/help`, `/quit`).

| Flag | Description |
|------|-------------|
| `--engine ENGINE` | Starting engine: `claude`, `codex`, `gemini`, `copilot`, `qwen`, `ollama`, `llama` (default: `claude`) |
| `--model MODEL` | Starting model (default: engine default) |
| `--execution-profile PROFILE` | `plan`, `safe`, or `yolo` (default: `plan`) |
| `--no-auto-context` | Disable startup auto-loading of `AGENTS.md` / `CLAUDE.md` context files |

### `maestro shell [options]`

Launch the interactive REPL. Slash commands include `/plan`, `/validate`, `/run`, `/suggest`, `/status`, `/explain`, `/last`, `/help`, and `/quit`. Autocomplete covers slash commands and YAML files in the current working directory.

| Flag | Description |
|------|-------------|
| `--plan PATH` | Path to the initial plan YAML file |

### `maestro skill [action] [options]`

Discover, search, and recommend available skills from `.claude/skills/`
directories.

| Flag | Description |
|------|-------------|
| `action` | `list` (default), `search`, or `recommend` |
| `--query`, `-q` | Search query for skill filtering |
| `--json` | Output as JSON |
| `--dir DIR` | Additional skill directories to scan (can repeat) |

`recommend` uses explicit frontmatter trigger metadata plus keyword overlap to
suggest which skill to use next and which small chain to follow when a skill
declares a recommended sequence.

### `maestro ci-analyze <run-path> [options]`

Analyze a failed CI run and suggest remediation actions (model escalation, timeout increase, retry configuration).

| Flag | Description |
|------|-------------|
| `run_path` | Path to a completed run directory |
| `--json` | Output analysis as JSON |

### `maestro budget [options]`

Show cross-run budget spending across discovered Maestro run roots.

| Flag | Description |
|------|-------------|
| `--root PATH` | Project root used to locate `.maestro-runs` directories (default: current working directory) |

---

## Custom Engine Plugins

Maestro discovers custom engines from the `maestro_cli.engines` entry-point group at
runtime. Built-in engines are registered internally and cannot be overridden via entry
points.

> This extension point is part of the frozen `1.x` public contract as of v1.31.0;
> see [V1_API_FREEZE.md](V1_API_FREEZE.md).

### Minimal contract

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

### Extension path

1. Publish a Python package that exposes `[project.entry-points."maestro_cli.engines"]`
2. Point the entry point at an `EnginePlugin` instance or zero-argument factory
3. Give the plugin a unique `name` that does not collide with built-in engines
4. Optionally supply `model_aliases`, `doctor_probe`, pricing hooks, and cost extraction
5. Install the package in the same environment as Maestro, then verify discovery with `maestro doctor`
