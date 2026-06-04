# Maestro Operations Manual — Agent Edition

You are an AI agent operating Maestro CLI. This is your operational bible.
Every decision rule here was paid for with real money and real failures.

---

## 1. Mental Model

Maestro is a DAG scheduler for AI tasks. You write YAML plans, Maestro runs them.

```
YAML plan → loader (validate) → PlanSpec → scheduler (DAG) → runners (subprocess per task) → artifacts
```

**You control**: plan design, engine/model selection, reliability strategy, cost.
**Maestro controls**: dependency resolution, parallel dispatch, retry logic, event logging.
**The engines control**: actual code generation (codex, claude, gemini, copilot, qwen, ollama, llama).

### Key abstractions

| Concept | What it is | Where it lives |
|---------|------------|----------------|
| `PlanSpec` | The validated plan | `models.py` |
| `TaskSpec` | One unit of work | `models.py` |
| `TaskResult` | Outcome of executing a task | `models.py` |
| `PlanRunResult` | Aggregate outcome of the whole plan | `models.py` |
| Engine | An AI CLI tool (codex, claude, etc.) | `runners.py` |
| DAG | Dependency graph of tasks | `scheduler.py` |

### Companion docs

- [PLAYBOOK.md](PLAYBOOK.md) — curated recipes for common tasks (9 recipes, cost data)

### Artifacts per run

```
.maestro-runs/<timestamp>_<plan>/
├── run_manifest.json    ← read this first (status, cost, tokens per task)
├── run_summary.md       ← human summary
├── events.jsonl         ← hash-chained event log
├── <task>.log           ← engine transcript
└── <task>.result.json   ← structured result
```

---

## 2. Plan Design

### Mandatory fields — EVERY plan

```yaml
version: 1
name: descriptive-name
max_cost_usd: 25.0           # ALWAYS. No ceiling = no safety net.
budget_warning_pct: 0.8       # ALWAYS. Get warned before you're broke.
max_parallel: 3               # ALWAYS. Limits API concurrency + cost.
goal: "One sentence of what this plan achieves"  # Free context injection into all engine tasks.

defaults:
  timeout_sec: 600            # ALWAYS. No timeout = tasks hang forever.
```

### Every task must have

```yaml
- id: unique-kebab-case
  description: "What this task does"       # ALWAYS. Shows in logs, TUI, reports.
  verify_command: ["py", "-c", "..."]      # ALWAYS on impl tasks. List format (not string).
  max_retries: 1                           # ALWAYS. 0 = fragile, no chance to self-correct.
```

### Tool restriction

For tasks consuming untrusted context, restrict tool access:

```yaml
- id: review-external
  engine: claude
  context_trust: untrusted
  allowed_tools: [Read, Grep, Glob]   # No write/shell access
  prompt: "Review the external submission"
```

Categories `read-only` and `no-shell` expand per engine. Claude uses
`--disallowedTools`; Codex maps to `--sandbox` levels; others use
system-prompt injection (advisory only).

### The one-task-one-file rule

Tasks that edit 2+ files frequently fail — the agent focuses on one and forgets the other.
Split into separate tasks, each with `Do NOT edit any other file` in the prompt.

### YAML anchors — DRY or die

```yaml
_impl: &impl
  engine: claude
  model: sonnet
  agent: python-developer
  edit_policy: efficient
  max_retries: 1

tasks:
  - id: task-a
    <<: *impl
    prompt: "..."
```

Separate anchors by role: `&impl`, `&test`, `&review`, `&docs`.

---

## 3. Engine & Model Selection

### Decision tree

```
Adding 1-2 fields to EXISTING dataclass?
  → claude haiku ($)
  → NO guard_command (haiku output isn't parseable)

Creating NEW code (dataclass, function, logic)?
  → claude sonnet + agent: python-developer ($$)
  → NEVER haiku for new code

Multi-file edit, algorithm, complex refactor?
  → claude sonnet + checkpoint: true ($$$)
  → OR codex@medium + checkpoint: true
  → --execution-profile yolo (not per-task args)

Writing tests?
  → claude sonnet + agent: qa-engineer ($$)
  → max_retries: 2, max_iterations: 5

Code review / QA pass?
  → Small scope (single file / interface review)?
    → claude sonnet + agent: code-reviewer
    → context_mode: raw, context_budget_tokens: 10000
    → preserves type signatures and exact diff lines
  → Large scope (cross-module audit)?
    → claude opus@medium + agent: code-reviewer ($$$)
    → context_mode: layered (NOT summarized — review needs the actual files,
      not a haiku rewrite)
    → judge: g_eval + rubric, cache: false

Security audit / architecture?
  → claude opus@high ($$$)
  → Only when genuinely needed

Local/free tasks?
  → ollama llama3 or codellama
  → llama llama3 or codellama
  → Zero cost, but less capable
```

### Model tiers

| Tier | Claude | Codex | Gemini | Copilot | Qwen | Ollama | Llama |
|------|--------|-------|--------|---------|------|--------|-------|
| Cheap | haiku | 5-mini / 5.4-mini | flash-lite | haiku | coder-turbo | phi3 | — |
| Standard | sonnet | 5.4@medium | flash | sonnet | coder | llama3 | llama3 |
| Powerful | opus@xhigh | 5.5@xhigh | pro | opus | max | mixtral | codellama |

`opus` resolves to Claude Opus 4.7 since 2026-04. The recommended starting
point for Opus 4.7 coding/agentic work is `xhigh` effort (per Anthropic's
own docs). Codex high tier moved 5.4 → 5.5 in the same window — 5.5 is more
expensive but materially more capable on long-horizon coding.

### Cost reality

| Engine + Model | Typical $/task | Use for |
|----------------|---------------|---------|
| Claude haiku | $0.05-0.15 | Trivial edits only |
| Claude sonnet | $0.50-1.50 | Standard implementation |
| Claude opus@high | $2.00-5.00 | Review, architecture |
| Claude opus@xhigh | $3.00-8.00 | Long-horizon agentic coding (>30 min loops) |
| Codex 5.4-mini@low | $0.10-0.40 | Cheapest serious Codex tier |
| Codex 5.4@medium | $0.50-1.50 | Standard implementation |
| Codex 5.5@xhigh | $1.50-6.00 | Hardest debugging, cross-module refactors |
| Ollama | $0.00 | Local work, zero cost |
| Llama | $0.00 | Local work via llama-cpp, zero cost |

Target: **< $1.00/task average**. If you're consistently above $2.00/task, you're using models that are too expensive for the work.

### Auto-routing

Set `model: auto` and Maestro picks based on complexity. Control bias with:
- `routing_strategy: cost_optimized` — pushes towards cheaper models
- `routing_strategy: quality_first` — pushes towards more capable
- `routing_strategy: balanced` — default, no bias

---

## 4. The Verification Stack

Five layers, cheapest first. Use as many as the task warrants.

```
1. verify_command  →  Deterministic. Exit 0 = pass. List format.     FREE
2. guard_command   →  Receives stdout via stdin. Exit 0 = pass.      FREE
3. assert:         →  Workspace assertions (file_contains, etc.)     FREE
4. judge (typed)   →  contains, regex, is-json, cost_under           FREE
5. judge (LLM)     →  llm-rubric, rubric (Likert), g_eval           COSTS $
```

### Critical rules

**verify_command**: ALWAYS list format `["py", "-c", "..."]`. String format uses cmd.exe on Windows → heredocs break.

**guard_command**: Receives engine STDOUT via stdin pipe. Good for `grep "expected" -` patterns. Does NOT receive file contents.

**assert:** (workspace assertions): Checks files on disk AFTER execution. Supports `file_contains`, `file_not_contains`, `file_regex`, `glob_exists`, `json_path_exists`, `composer_package_present`, `npm_package_present`.

**judge typed assertions**: `contains`/`regex` check ENGINE STDOUT — NOT files, NOT verify output.

### THE #1 PITFALL: judge contains/regex on engine tasks

**NEVER** use `type: contains` or `type: regex` in judge criteria on engine tasks.
They search the engine's stdout (JSON output), not the generated files.
This is the single biggest source of false failures — cost $50+ across real plans.

Use instead:
- `verify_command` + exit code for deterministic checks
- `guard_command` for stdout pattern matching
- `assert:` for file content checks
- `type: llm-rubric` or `type: rubric` for subjective quality
- `type: is-json`, `type: cost_under`, `type: duration_under` for other deterministic checks

### Judge configuration for review tasks

```yaml
judge:
  method: g_eval               # Two-phase (generate steps → score). More consistent.
  aggregation: weighted_mean   # Or min (all criteria must pass individually)
  # timeout_sec: omitted → auto-scaled to 120s+ based on criteria count (W22)
  criteria:
    - type: rubric
      name: correctness
      weight: 2.0
      min_score: 3
      levels:
        - { score: 1, description: "Completely wrong" }
        - { score: 3, description: "Mostly correct with minor issues" }
        - { score: 5, description: "Correct and well-tested" }
    - type: cost_under
      value: 5.0
  pass_threshold: 0.7
  on_fail: warn                # warn for reviews, retry for impl
```

> **Judge timeout auto-scaling**: When `timeout_sec` is omitted, Maestro computes a sensible default: `g_eval` = 120s, `debate` = rounds×120s, +15s per criterion over 4, ×quorum. W22 warns when explicit values are too low. See [PLAN_GUIDE.md — Judge Timeout Auto-Scaling](PLAN_GUIDE.md#judge-timeout-auto-scaling).

#### Judge Advanced Configuration

**Timeout auto-scaling** (W22): when `judge.timeout_sec` is omitted, Maestro auto-scales:
- `direct`: 60s base
- `g_eval`: 120s base
- `debate`: rounds × 120s
- +15s per criterion over 4
- ×quorum multiplier

W22 warns when explicit `timeout_sec` is below the recommended minimum.

**Quorum**: `judge.quorum: N` (N >= 2) runs N independent judge evaluations. Strategies: `majority` (default — more than half pass), `unanimous` (all pass), `any` (at least one passes). Score averaged across valid evaluations.

**Presets**: `judge.preset` provides pre-built criteria sets:
- `code_quality`: maintainability, readability, error handling
- `security_audit`: injection, auth, data exposure
- `ai_slop_detection`: detects lazy/generic AI output
- `cwe_injection`: SQL/Command/XSS/Path Traversal (CWE-89/78/79/22) — `aggregation: min`, threshold 0.8
- `cwe_auth`: Authentication/Access Control/Credentials/Sessions (CWE-287/284/256/384) — `aggregation: min`, threshold 0.8
- `cwe_data_exposure`: Data Exposure/Crypto/Error Leakage (CWE-200/327/209) — `aggregation: min`, threshold 0.7
- `cwe_top_25`: Broad OWASP coverage (5 rubrics) — `aggregation: min`, threshold 0.75

**When to choose**: Use CWE profiles for targeted vulnerability scanning (e.g., after a dependency audit). Use `security_audit` for general security review. CWE profiles use `aggregation: min` — every criterion must individually pass.

Explicit YAML values override preset defaults.

**Debate method**: `judge.method: debate` runs adversarial evaluation with configurable `judge.debate_rounds` (default 2).

**Comparative evaluation**: on judge retry (`on_fail: retry`), the next attempt is compared against the previous for relative quality assessment.

---

## 5. Context Strategy

### When to use each mode

| Mode | Cost | When | Template var |
|------|------|------|-------------|
| `raw` | Free | Few upstreams, small output | `{{ task.stdout_tail }}` |
| `layered` | Free | 3+ upstreams, budget-aware L0/L1/L2 tiers (40-65% savings) | `{{ task.stdout_tail }}` |
| `summarized` | 1 haiku/upstream | Large outputs, need key facts | `{{ task.summary }}` |
| `map_reduce` | N haiku + 1 synth | 3+ upstreams, unified view | `{{ upstream_synthesis }}` |
| `recursive` | Index + extract + brief | Full workspace awareness | `{{ workspace_brief }}` |

### context_model — decouple compression model from execution model

The model used for summarization/extraction/brief operations (haiku by default) can be overridden independently of the task's execution model:

```yaml
defaults:
  claude:
    model: sonnet
    context_model: haiku     # cheap model for context ops (default)

tasks:
  - id: expensive-task
    engine: claude
    model: opus
    context_model: flash     # use Gemini Flash for context ops, opus for execution
    context_mode: recursive
```

Priority: `task.context_model` > `defaults.<engine>.context_model` > `haiku`

### Budget rules

- **ALWAYS set `context_budget_tokens`** on tasks with `context_from`
- Normal tasks: 4000-6000 tokens
- Review tasks: 8000-10000 tokens
- Recursive context tasks: 6000-8000 tokens
- When budget is exceeded, eviction uses RRF fusion (BM25 keyword relevance + DAG hop distance) to keep the most relevant upstreams; order `depends_on` by semantic relevance for best results

### Context variables available in prompts

```
{{ task-id.status }}        — success/failed/soft_failed/skipped
{{ task-id.exit_code }}     — process exit code
{{ task-id.stdout_tail }}   — last N lines of stdout
{{ task-id.log }}           — full path to .log file
{{ task-id.duration }}      — seconds
{{ task-id.files_changed }} — auto-extracted file list (structured, zero cost)
{{ task-id.errors }}        — auto-extracted error lines (structured, zero cost)
{{ task-id.warnings }}      — auto-extracted warning lines (structured, zero cost)
{{ task-id.decisions }}     — auto-extracted decision lines (structured, zero cost)
{{ task-id.result_text }}   — auto-extracted result summary (structured, zero cost)
{{ task-id.summary }}       — haiku summary (needs summarized mode)
{{ upstream_synthesis }}    — combined synthesis (needs map_reduce)
{{ workspace_brief }}       — workspace context doc (needs recursive)
{{ task-id.output.FIELD }}  — structured output field (needs output_schema on upstream)
{{ workspace_root }}        — resolved workspace path
{{ goal }}                  — plan-level goal string
{{ plan_name }}             — plan name
{{ task_id }}               — current task ID
{{ matrix.KEY }}            — matrix expansion variable
{{ batch.item }}            — current batch item (needs batch: mode)
{{ task_knowledge }}        — cross-run knowledge (auto-injected, zero config, zero cost)
{{ contracts_summary }}     — summary of all consumed contracts
{{ consistency_summary }}   — summary of all consistency groups
{{ watch.blame }}           — JSON blame analysis from target plan's last run (needs blame_plan)
{{ watch.manifest }}        — compact task status summary from target plan's last run (needs blame_plan)
{{ watch.lessons }}         — formatted lessons from knowledge archive (time-decayed)
{{ watch.consolidated }}    — consolidation agent output
{{ improve.frozen_tasks }}  — comma-separated list of frozen task IDs
```

---

## 6. Resilience Features

### Auto-escalation

```yaml
escalation: [haiku, sonnet, opus]
max_retries: 2
```

Each retry bumps to the next model tier. Start cheap, escalate on failure.

### Cross-engine fallback

```yaml
engine: claude
fallback_engine: codex
fallback_model: "5.4"
```

Triggers on infrastructure failures ONLY (CLI missing, API down, rate limits).
**NEVER use with engine-specific `args:`** — fallback inherits args and crashes.

### Circuit breaker

```yaml
circuit_breaker:
  max_failures: 3
  reset_after_sec: 300
```

Trips after N consecutive failures. Prevents burning budget on persistently broken tasks.

### Retry strategies

| Strategy | Delay pattern | Use when |
|----------|--------------|----------|
| `constant` | Same delay every time | Default |
| `linear` | base × attempt | Gradual backoff |
| `exponential` | base × 2^attempt | Rate limit recovery |

### Checkpoint protocol

```yaml
checkpoint: true
max_retries: 2
```

For long-running tasks. Creates `MAESTRO_CHECKPOINT_DIR`. On retry, checkpoint data is auto-injected so the agent can continue from where it left off.

---

## 7. Operations Playbook

### Starting a new plan

1. **Scaffold with sane baselines** — `maestro scaffold brief.yaml --strict-defaults` injects `defaults.timeout_sec=1500`, `defaults.retry_delay_sec=[60, 120]`, `max_cost_usd=10.0`, `budget_warning_pct=0.8`. Pre-empts the warnings the rest of this section would otherwise tell you to fix.
2. Refine the DAG (tasks, dependencies, engines, prompts).
3. **Run `maestro check plan.yaml --strict`** — bundles validate + audit, exits non-zero on warnings (CI-grade gate). Replaced `maestro validate plan.yaml` as the canonical pre-run check on 2026-04-27. Covers E001-E072 *and* SEC001-SEC023.
4. **First run with `max_retries: 0`** if you genuinely don't want retries (smoke tests, creative-generation tasks). W20 won't fire — `max_retries: 0` is a valid silencer. Otherwise pick at least one W20 escape valve (see Section 11 verification rules).
5. Tell the user to run: `maestro run plan.yaml --execution-profile yolo --output live`.

> **Don't whack the warning.** When a warning fires, look up its escape valve and pick one — don't cycle through `max_retries`, `retry_delay_sec`, `verify_command` settings hoping one combination silences it. An internal post-mortem (2026-04-26) documents an AI agent doing exactly that for most of its authoring iterations. Each warning has *one* concept behind it; the choice is which valve fits your failure mode.

### NEVER run maestro from Claude Code's Bash tool

`subprocess.run(timeout=X)` doesn't kill engine child processes on Windows.
Result: orphaned processes, blocked scheduler, wasted resources.
**ALWAYS give the user the command to run in their terminal.**

### Diagnosing a failed run

1. Read `run_manifest.json` — find which tasks failed
2. Read `<task>.result.json` — check `status`, `exit_code`, `message`
3. Read `<task>.log` — the engine transcript shows what happened
4. Check `events.jsonl` — timeline of events, escalations, fallbacks
5. Run `maestro blame <run-path>` — automated causal analysis

### Common failure patterns

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| exit_code 127 | Tool not on PATH | Use `grep` not `rg`. Check `maestro doctor` |
| exit_code 124 | Timeout | Increase `timeout_sec` (1200s+ for complex tasks) |
| exit_code 3 (Claude) | Often NOT a real failure | Check JSON `is_error` field. May need `allow_failure: true` |
| "rate limit" / "hit your limit" | API quota exhausted | Lower `max_parallel`, add `retry_delay_sec`, wait |
| verify passes but judge fails | Judge checking wrong thing | Remove `contains`/`regex` from judge. Use `llm-rubric` |
| Judge times out (task OK) | `timeout_sec` too low for method/criteria | Omit `timeout_sec` (auto-scales) or set >= W22 recommended minimum |
| Task skipped | Dependency failed | Fix upstream task first, then `--resume-last` |
| "Budget exceeded" | `max_cost_usd` hit | Increase budget or optimize model selection |
| `honeypot_triggered` event | Injected instructions detected in task output | Review `context_trust` settings; task may be processing malicious input |
| `scope_violation` event | Task modified files outside declared `output_scope` | Check scope globs; add missing patterns or restrict task prompt |
| `worktree_verification` failed | Agent claims don't match git diff | Review `unclaimed_files` and `phantom_files` in event; may indicate hallucination |

### Resuming from failure

```bash
maestro run plan.yaml --resume-last    # Resume from most recent run
maestro run plan.yaml --resume .maestro-runs/20260314_plan/  # Resume specific run
```

Only re-runs failed/skipped tasks. Succeeded tasks are skipped (cached).

### Mid-Task Signals

Enable with `signals: true` on a task (or in `defaults.<engine>.signals`). The executing agent writes `[MAESTRO_SIGNAL] {"type":"..."}` lines to stdout.

**Signal types**: `progress` (percentage), `metric` (named value), `log` (message), `artifact` (file path), `timeout_extend` (request more time), `budget_query` (check remaining budget), `checkpoint` (save progress).

**Key use cases**:
- Long tasks (>30min): `timeout_extend` prevents premature timeout
- Watch loops: `metric` signals feed metric extraction
- Resilience: `checkpoint` enables recovery on retry

**Limits**: 10 signals/sec, 1000 total, 4KB max per signal.

### Watch mode (autonomous iteration)

**Custom mode** (metric-driven):
```yaml
watch:
  mode: custom          # default
  metric: test_coverage
  metric_direction: higher_is_better
  metric_source: stdout_regex
  metric_pattern: "coverage: (\\d+\\.?\\d*)%"
  max_iterations: 10
  on_regression: rollback
  plateau_threshold: 3
  plateau_action: stop
```

**Improve mode** (built-in plan improvement loop — no metric config needed):
```yaml
watch:
  mode: improve         # auto-sets metric=tasks_passed, on_regression=rollback
  workspace_root: /path/to/project
  max_iterations: 10
  improve_model: sonnet
  max_cost_usd: 20.0
```

Watch template vars: `{{ watch.iteration }}`, `{{ watch.best_metric }}`, `{{ watch.last_metric }}`, `{{ watch.history }}`, `{{ watch.program }}`, `{{ watch.lessons }}` (knowledge archive), `{{ watch.consolidated }}` (consolidation), `{{ watch.blame }}` (blame JSON, needs `blame_plan`), `{{ watch.manifest }}` (task status summary, needs `blame_plan`).

### Workflow Libraries

Use built-in workflow libraries for common task patterns:

```bash
maestro scaffold --list-libraries          # see available libraries
maestro scaffold brief.yaml --library rest-api  # use a library
```

5 built-in: `rest-api`, `refactor`, `security-review`, `bug-fix`, `test-backfill`.
Brief tasks override library tasks by matching ID. New tasks are appended.
See [PLAYBOOK.md](PLAYBOOK.md) for detailed recipes.

---

## 8. Policies

Runtime guardrails that catch bad decisions at dispatch time.

```yaml
policies:
  - name: no-opus-without-judge
    rule: "model == 'opus' and not has_judge"
    action: block
    message: "Opus tasks must have quality gates"

  - name: require-verify
    rule: "engine != None and verify_command == None"
    action: warn
    message: "Engine tasks should have verify_command"
```

Available fields for rules:
- **Task**: `id`, `engine`, `model`, `tags`, `timeout_sec`, `max_retries`, `allow_failure`, `requires_approval`, `cache`, `description`, `cost_usd`, `has_judge`, `execution_profile`, `context_trust`, `allowed_tools`, `has_allowed_tools`, `dynamic_group`, `contract_type`, `has_consistency_group` (non-exhaustive — see `policy.py` `_SAFE_TASK_FIELDS`)
- **Plan**: `name`, `max_cost_usd`, `max_parallel`, `execution_profile`, `fail_fast`
- **Operators**: `==`, `!=`, `<`, `>`, `<=`, `>=`, `and`, `or`, `not`, `in`, `not in`

---

## 9. Security

### Before running any plan

```bash
maestro audit plan.yaml --fix       # Scan SEC001-SEC023 + auto-fix what's possible
maestro audit plan.yaml --coverage  # Per-category security coverage breakdown
```

Key audit rules:
- **SEC001**: No budget → auto-adds `max_cost_usd: 10.0`
- **SEC002**: Secrets in prompts → manual fix needed
- **SEC003**: Secrets in env → auto-adds `secrets: auto`
- **SEC004**: Yolo/bypass flags without justification
- **SEC008**: Destructive commands without approval gates
- **SEC015**: `when:` references unbounded upstream fields (`stdout_tail`, `log`)
- **SEC016**: `context_from` pulls raw engine output without `guard_command` validation. Refined 2026-04-26 to fire **only on `context_mode: raw`** — `summarized` / `map_reduce` / `recursive` / `layered` / `selective` / `structural` / `council` / `knowledge_graph` are exempt (LLM mediation or heuristic extraction provides partial injection resistance). Switching the consumer's mode is now a valid alternative to adding `guard_command`.
- **SEC017**: `context_from` references tasks with external data but no `context_trust` set
- **SEC018**: Task inherits tainted context from upstream without `guard_command`/`verify_command`
- **SEC019**: `context_trust: untrusted` without `honeypot: true` for injection detection
- **SEC020**: Upstream PII-like references without `output_redact` patterns
- **SEC021**: Destructive commands without `phantom_workspace` or `requires_approval`
- **SEC022**: Contract consumer without `verify_command`/`guard_command`
- **SEC023**: Untrusted context without `allowed_tools` restriction

`--coverage` maps rules to 9 risk categories (Agent-Tool Coupling, Data Leakage, Injection, Identity/Provenance, Memory Poisoning, Non-Determinism, Trust Exploitation, Timing/Monitoring, Workflow Architecture). Use it to identify gaps.

### Secrets

```yaml
secrets: auto                    # Auto-detect by name pattern (KEY, SECRET, TOKEN)
# OR
secrets: [OPENAI_API_KEY, DB_PASSWORD]   # Explicit list
```

Always use `--mask-secrets` when running plans with API keys in the environment.

### Untrusted Context Detection

Mark tasks consuming external or untrusted data with `context_trust: untrusted`:

```yaml
- id: fetch-external
  command: "curl -s https://api.example.com/data > output.json"
  context_trust: untrusted

- id: process
  engine: claude
  context_from: [fetch-external]
  guard_command: "python validate.py"
  prompt: "Analyze results"
```

**Behaviour**: injection patterns stripped, output sandboxed in `<observation>` tags, taint propagates to downstream consumers lacking `guard_command` or `verify_command`.

**Events**: `taint_detected` emitted at run start for all tainted tasks.

**Policy integration**: `context_trust` available as task field in policy rules.

### Output Scope Validation

Restrict task file modifications with `output_scope`:

```yaml
- id: auth-fix
  output_scope: ["src/auth/**", "tests/auth/**"]
```

After completion, scope violations are logged via `scope_violation` event. The `OutputEnvelope` on TaskResult captures the output hash (SHA-256) and any violations.

### Honeypot Injection Detection

For tasks consuming untrusted context, enable honeypot decoys:

```yaml
- id: process-user-input
  context_trust: untrusted
  honeypot: true
```

Injects trap values (fake API keys, URLs) into the context. If the agent accesses these decoys, a `honeypot_triggered` event is emitted — indicating prompt injection. SEC019 warns when honeypot is missing on untrusted tasks.

---

## 10. Hard-Won Rules (key pitfalls, see PITFALLS.md for the full catalogue P1-P39)

These rules cost real money to learn. Violating any one can waste an entire run budget.

### Plan structure

| Rule | Prevents | Cost of violation |
|------|----------|------------------|
| `max_cost_usd` on EVERY plan | Uncapped spending | $15.30 (P10) |
| `description:` not `name:` on tasks | Silent field ignore | 13 tasks invisible (P21) |
| One task = one file | Agent forgets second file | Multiple retries wasted (E3) |
| YAML anchors for shared defaults | Typo propagation | N tasks × M retries (E5) |

### Engine flags

| Rule | Prevents | Cost of violation |
|------|----------|------------------|
| Test engine CLI flags locally FIRST | Wrong flags burn all retries | $13.50 (P17) |
| Use `--execution-profile yolo` not per-task args | Fallback inherits bad args | $3.76 (P15) |
| Codex CREATE needs `--dangerously-bypass-approvals-and-sandbox` | `--full-auto` can't create files | $9.74 (P16) |
| `grep` not `rg` in verify/guard | `rg` not on subprocess PATH | $14.91 (P20) |

### Verification

| Rule | Prevents | Cost of violation |
|------|----------|------------------|
| **NEVER** `contains`/`regex` in judge on engine tasks | Checks stdout not files | $50+ across plans |
| List format `["py", "-c", "..."]` for verify | cmd.exe breaks heredocs | $0.16 + 12 tasks skipped (P8) |
| `max_retries: 0` on first run of new plan | Burning retries on broken flags; W20 won't fire | $13.50 (P17) |
| If `max_retries > 0`, give it ONE W20 escape valve | Retries reproduce identical conditions and fail the same way | Repeated authoring iterations (internal post-mortem) |
| W20 escape valves: `verify_command` / `guard_command` / `assert` / `judge` / `escalation` / `fallback_engine` / list-form `retry_delay_sec` | Pick one that matches the failure mode you expect; don't stack all of them | Warning whack-a-mole |
| `max_iterations` on `on_fail: retry` tasks | Infinite retry spiral | Unbounded cost |
| Omit `judge.timeout_sec` (let auto-scale) or set >= W22 minimum | Judge times out, task marked failed | $1.80+ wasted per occurrence (P22) |
| Sonnet+ for synthesis tasks | Haiku asks questions instead of working | Task failure (P23) |
| Specific constraints, not generic personas | Generic expert personas help alignment but hurt precision on coding/knowledge tasks; route them selectively | Wrong output (PRISM, 2026) |
| `tsc --noEmit` (or equivalent) on multi-file TS plans | Integration bugs invisible to per-file checks | Costly retry loop + manual fix (P32) |
| Review tasks: `on_fail: fail` when correctness matters | Reviewer says FAIL, run says SUCCESS | Broken deliverable (P33) |
| Cross-file integration check after each wave | Interface mismatches between producer/consumer | Several integration bugs in a large multi-task run (P32) |
| `context_mode: raw` for interface-consuming tasks | Lossy compression drops type definitions | Category A bugs (P32) |

### Cost control

| Rule | Prevents | Cost of violation |
|------|----------|------------------|
| Lower `max_parallel` for matrix batches | Rate limit cascade | 6 tasks failed (P3) |
| `allow_failure: true` only on non-critical tasks | Quality gate bypass | Zero quality signal (P22) |
| 1200s+ timeout for complex generation | Timeout retries (600s × 3) | 1800s wasted (P5) |

---

## 11. Pre-Flight Checklist

Run through this EVERY TIME before `maestro run`. No exceptions.

### Structure
- [ ] `max_cost_usd` defined
- [ ] `budget_warning_pct` defined
- [ ] `goal:` defined (free context injection)
- [ ] `defaults.timeout_sec` defined
- [ ] `description` on ALL tasks
- [ ] YAML anchors for shared defaults

### Verification
- [ ] ALL impl tasks have `verify_command` (list format)
- [ ] `max_retries: 1`+ on tasks with verify
- [ ] `on_fail: retry` tasks have `max_iterations`
- [ ] **ZERO** `type: contains`/`regex` in judge of engine tasks
- [ ] `assert:` rules use only known fields (E018 rejects unknown fields like `negate`)
- [ ] Negative assertions use `file_not_contains` / `file_regex_absent` (not `negate: true`)
- [ ] `grep` not `rg` in verify/guard commands
- [ ] verify/guard commands tested locally

### Engine flags
- [ ] Engine CLI flags tested locally
- [ ] Codex CREATE tasks use `--dangerously-bypass-approvals-and-sandbox`
- [ ] No `fallback_engine` on tasks with engine-specific `args:`
- [ ] First run of new plan: `max_retries: 0`

### Quality
- [ ] Review task uses `judge.method: g_eval` or rubric with `model: sonnet`+
- [ ] Judge `timeout_sec` either omitted (auto-scaled) or set >= recommended minimum (W22)
- [ ] `g_eval` with 5+ criteria: verify auto-scaled timeout is adequate (120s base + 15s/extra criterion)
- [ ] Judge `quorum` ≤ 3 (W24 — consensus degrades beyond 3 evaluators)
- [ ] Review/QA tasks have `cache: false`
- [ ] Review/QA tasks use `on_fail: fail` (not `warn`) when correctness matters
- [ ] Each task edits NO MORE THAN 1 file
- [ ] Haiku only for trivial edits, NEVER for creating new code
- [ ] Synthesis tasks use sonnet+
- [ ] `allow_failure: true` used surgically (not on everything)

### Integration (multi-file codegen)
- [ ] Shared types/protocol file created as first wave (single source of truth)
- [ ] Interface-consuming tasks use `context_mode: raw` (not `layered`/`summarized`)
- [ ] `tsc --noEmit` (or equivalent compiler) as verify_command on each wave
- [ ] At least one cross-file integration check task after implementation
- [ ] All HTTP response/request shapes specified explicitly in prompts
- [ ] Constructor signatures specified explicitly when multiple classes share an interface

### Context
- [ ] Tasks with `context_from` have `context_budget_tokens`
- [ ] `context_from` entries are in `depends_on`
- [ ] Recursive context tasks have `workspace_root`
- [ ] `output_schema` tasks: downstream consumers have `depends_on` + `context_from`
- [ ] `--set` template vars use stable values (no timestamps/UUIDs — breaks caching)

### Flow
- [ ] Release task has `when:` conditional on review success
- [ ] `escalation` and `fallback_engine` considered
- [ ] `tags:` on tasks for surgical reruns
- [ ] `dynamic_group` tasks have `engine` + `output_schema` (E063)
- [ ] `dynamic_group` not combined with `group`/`batch`/`matrix` (E064)

### Security & Signals
- [ ] `context_trust` set on tasks consuming external/untrusted data
- [ ] `control_flow_integrity: true` on plans with untrusted context chains
- [ ] `signals: true` on long-running tasks (>1800s) for checkpoint/progress
- [ ] Judge `timeout_sec` omitted (auto-scales) or >= W22 minimum for method/criteria count
- [ ] `output_schema` tasks have JSON-producing prompts
- [ ] `output_scope` declared for tasks that should only modify specific files
- [ ] `honeypot: true` on tasks consuming `context_trust: untrusted` context
- [ ] `allowed_tools` set on engine tasks with untrusted context (SEC023)
- [ ] Council `graph` topology: `connections` map covers all participant roles (E072)
- [ ] CWE judge preset selected for security-critical tasks (`cwe_injection`, `cwe_auth`, etc.)
- [ ] Workflow library considered for common patterns (`maestro scaffold --list-libraries`)

### Final
- [ ] `maestro check plan.yaml --strict` passes (validate + audit, exits non-zero on warnings; covers W20-W30 + SEC001-SEC023 in one pass)
- [ ] If W20 fired, picked an explicit escape valve (`verify_command` / `guard_command` / `assert` / `judge` / `escalation` / `fallback_engine` / positive `retry_delay_sec`) or accepted the design with `max_retries: 0`
- [ ] If SEC016 fired, either added `guard_command` upstream OR switched downstream `context_mode` to a non-`raw` mode — whichever fits the trust boundary
- [ ] If `--with-suggest` produced optimization hints from prior runs, considered them
- [ ] Give user the run command (NEVER run from Claude Code Bash tool)

---

## 12. Quick Reference: Error Codes

### Validation (E001-E072)
| Code | Meaning |
|------|---------|
| E001 | Missing version field |
| E002 | Version must be 1 |
| E003 | Duplicate task ID |
| E004 | Circular dependency |
| E005 | Unknown dependency reference |
| E006 | Invalid engine name |
| E007 | Missing prompt source |
| E008 | Invalid field value (reasoning_effort, edit_policy, etc.) |
| E009 | Invalid model name |
| E010 | context_from not in depends_on |
| E018 | Type mismatch / unknown fields in `assert:` rules or judge criteria |
| E019 | Context budget range |
| E020 | Judge config (timeout_sec < 10, unknown criteria fields) |
| E021 | Recursive without workspace_root |
| E022 | max_iterations out of range |
| E023 | budget_warning_pct out of range |
| E024-E028 | Import validation |
| E029 | approval_message without requires_approval |
| E030 | Invalid escalation list |
| E031 | Invalid fallback config |
| E032-E044 | Watch validation |
| E045-E046 | Worktree validation |
| E050 | Circuit breaker config |
| E051 | Invalid retry_strategy |
| E052 | Invalid policy config |
| E053 | Invalid routing_strategy |
| E054 | Invalid judge.quorum (must be >= 2) |
| E055 | Invalid judge.quorum_strategy |
| E056 | quorum_strategy without quorum |
| E057 | Invalid batch config (missing items/template) |
| E058 | batch.max_per_call < 1 |
| E060 | batch on command/group task (engine only) |
| E062 | batch and matrix mutually exclusive |
| E063 | dynamic_group requires engine + output_schema |
| E064 | dynamic_group conflicts with group/batch/matrix |
| E065 | Invalid `context_trust` value (must be `trusted` or `untrusted`) |
| E066 | Invalid `watch.max_total_steps` (must be >= 1) |
| E067 | Invalid `reminders` configuration |
| E068 | Invalid `context_compaction` value |
| E069 | Invalid MCP server configuration |
| E070 | Unknown MCP server reference in `mcp_tools` |
| E071 | `allowed_tools` on command/group task (engine only) |
| E072 | Invalid council graph topology `connections` |

### Runtime (E100-E110)
| Code | Meaning |
|------|---------|
| E100 | Prompt file not found |
| E101 | Markdown heading not found |
| E102 | Unsupported engine |
| E103 | No engine specified |
| E104 | Workdir resolution failed |
| E105 | Command build failure |
| E106 | Group sub-plan not found or failed to load |
| E107 | Judge execution error |
| E108 | Workspace index build failure |
| E109 | Workspace extraction LLM error |
| E110 | Workspace brief LLM error |

---

## 13. Command Quick Reference

```bash
# Core workflow
maestro scaffold brief.yaml --strict-defaults  # First-run plan with sane baselines
maestro check plan.yaml                        # validate + audit, single exit code
maestro check plan.yaml --strict               #   same, but exits non-zero on warnings
maestro check plan.yaml --with-suggest --json  #   include optimization hints, JSON report
maestro validate plan.yaml                     # legacy (single-step validation only)
maestro run plan.yaml --execution-profile yolo --output live
maestro run plan.yaml --resume-last
maestro replan plan.yaml --max-attempts 3

# Diagnosis
maestro doctor --json
maestro blame .maestro-runs/run_dir/
maestro verify .maestro-runs/run_dir/
maestro audit plan.yaml --fix --json
maestro audit plan.yaml --coverage

# Analysis
maestro diff run_a run_b --json
maestro suggest plan.yaml --json
maestro explain plan.yaml --json
maestro status plan.yaml --json
maestro eval eval.yaml run_dir --json

# Interactive
maestro chat --engine claude --model sonnet
maestro shell

# Autonomous iteration
maestro watch plan.yaml --output live
maestro watch plan.yaml --resume-last

# Protocols
maestro mcp-server                    # Launch MCP protocol server (stdio transport)

# Maintenance
maestro cleanup plan.yaml --keep 5
maestro backfill-costs
maestro report .maestro-runs/run_dir/

# Skills + budget + ci-analyze (less common)
maestro skill list
maestro budget
maestro ci-analyze .maestro-runs/run_dir/
```
