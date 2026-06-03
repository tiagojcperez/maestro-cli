# Maestro CLI — Playbook

Curated recipes for common development tasks. Each recipe includes the
YAML pattern, key decisions, pitfalls to avoid, and expected cost.

Based on 42+ real production runs totalling ~$306 in engine costs.
12 recipes covering feature implementation through multi-model council deliberation.

**Audience**: Developers authoring Maestro plans, AI agents creating plans
via scaffold, and anyone optimising existing plans.

**Companion docs**:
- [PLAN_GUIDE.md](PLAN_GUIDE.md) — schema reference
- [AGENT_OPS.md](AGENT_OPS.md) — operations manual
- [PITFALLS.md](PITFALLS.md) — full pitfall catalogue
- [MODELS.md](MODELS.md) — engine/model tables

---

## Quick Reference: Engine Selection

| Task Type | Engine + Model | Cost/Task | When |
|-----------|---------------|-----------|------|
| Trivial fix (typo, config) | `claude` + `haiku` | $0.05-0.15 | Single file, obvious change |
| Standard implementation | `claude` + `sonnet` | $0.50-1.50 | CRUD, views, services |
| Complex implementation | `codex` + `5.4@medium` | $0.50-1.50 | Algorithms, state machines |
| Frontier coding / long-horizon agentic | `codex` + `5.5@xhigh` or `claude` + `opus@xhigh` | $1.50-6.00 | Cross-module refactors, > 30 min agentic loops, hardest debugging |
| Code review | `claude` + `sonnet` | $0.30-0.80 | Always use `agent: code-reviewer` |
| Security audit | `claude` + `opus@high` | $2.00-5.00 | Auth, crypto, injection |
| Quick validation | `ollama` + `llama3` | $0.00 | Local checks, no API cost |
| Budget-sensitive | `codex` + `5.4-mini@low` | $0.10-0.40 | When cost matters most (5.4-mini is ~3× cheaper than 5.4) |

**Rule of thumb**: Start with the cheapest model that could work. Escalate
only on failure. The `escalation` field automates this.

---

## Shape Plans to Maestro's Grain

> **"Plan shape is the single biggest predictor of how Maestro will behave."**
> — an internal post-mortem

After several same-day runs by the same author, same engine, same project, the
cost spread was large — driven entirely by plan shape. Tasks-with-LLM-context-from-LLM
(Plan #1) cost noticeably more than tasks-with-disk-from-disk (Plan #2), which also
saved substantial parallelism.

Two principles that fall out:

1. **Prefer disk-based handoffs over `context_from` between engine tasks.** When
   an upstream task produces a concrete file, instruct the downstream task to
   `Read` it directly via prompt instructions. This avoids stdout_tail
   truncation, sidesteps SEC016 entirely, and removes context-budget juggling.
   Reach for LLM-to-LLM `context_from` only when the synthesis genuinely cannot
   be deferred to disk (see Recipe 8 for the cases where it is unavoidable).

2. **Prefer deterministic verification over LLM judges.** A `verify_command`
   like `pytest tests/X.py -q` is exit-code-binary, free, and unambiguous. An
   LLM judge costs $0.05-0.30 per call and introduces verdict variance. Use
   judges for things only an LLM can evaluate (rubric-based review, prose
   quality, design feedback); use deterministic checks everywhere else.

The recipes below are organised so that **Recipe 4 (Test Backfill)** is the
canonical "ideal-fit" shape — homogeneous tasks, per-task disk reads, pytest
verify — and **Recipe 8 (TS Codegen)** is the canonical "unavoidably chained"
shape that you reach for when the work cannot be parallelised structurally.

Run `maestro check <plan.yaml>` before any first run; it bundles
`validate` + `audit` and surfaces the warnings that catch shape-mismatches
early. For a fresh plan, `maestro scaffold <brief.yaml> --strict-defaults`
seeds sane baselines (timeout=1500, retry_delay=[60,120], max_cost_usd=10)
that pre-empt the most common authoring warnings.

---

## Recipe 1: Feature Implementation

The most common pattern — implement a feature with quality gates.

```yaml
version: 1
name: add-user-auth
workspace_root: "C:/projects/my-app"
max_cost_usd: 5.0
fail_fast: true

# DRY defaults with YAML anchors
_impl: &impl
  engine: claude
  model: sonnet
  max_retries: 1
  timeout_sec: 600

tasks:
  - id: implement
    <<: *impl
    prompt: |
      Add JWT authentication to the Express.js API.
      - POST /auth/login — accept email+password, return JWT
      - Middleware to verify JWT on protected routes
      - Use bcrypt for password hashing
    verify_command: "node -e \"require('./src/auth')\""
    tags: [implementation]

  - id: write-tests
    <<: *impl
    depends_on: [implement]
    agent: qa-engineer
    prompt: |
      Write tests for the authentication module.
      Test: login success, login failure, JWT verification, expired tokens.
    verify_command: "npm test -- --grep auth"
    tags: [testing]

  - id: review
    engine: claude
    model: sonnet
    agent: code-reviewer
    depends_on: [implement, write-tests]
    context_from: ["*"]
    context_mode: summarized
    prompt: |
      Review the authentication implementation and tests.
      Check: SQL injection, timing attacks, token storage, error leakage.
    judge:
      preset: security_audit
    tags: [review]
```

**Key decisions**:
- `verify_command` on implementation tasks catches syntax errors at zero cost
- `context_mode: summarized` gives the reviewer a digest, not raw output
- `judge.preset: security_audit` enforces security standards automatically
- YAML anchors (`&impl`) keep the plan DRY

**Pitfalls to avoid**:
- P1: Use `|` for multi-line prompts, never `>` (collapses newlines)
- P23: Be specific in prompts — "Add JWT auth" not "Improve security"
- P22: Don't use `allow_failure: true` on implementation tasks

**Expected cost**: $1.50-3.00 for 3 tasks

---

## Recipe 2: Code Refactoring

Analyse → implement → verify pattern. Critical: the analysis must be
separate so the implementation has a concrete plan.

```yaml
version: 1
name: refactor-data-layer
workspace_root: "C:/projects/my-app"
max_cost_usd: 8.0

_defaults: &defaults
  engine: claude
  timeout_sec: 600

tasks:
  - id: analyse
    <<: *defaults
    model: sonnet
    agent: code-reviewer
    prompt: |
      Analyse src/data/ for refactoring opportunities.
      Focus on: duplication, inconsistent patterns, missing error handling.
      Output a structured JSON change plan:
      {"files": [{"path": "...", "changes": [...]}]}
    tags: [analysis]

  - id: refactor
    <<: *defaults
    model: sonnet
    depends_on: [analyse]
    context_from: [analyse]
    max_retries: 1
    escalation: [sonnet, opus]
    prompt: |
      Apply the refactoring changes from the analysis.
      {{ analyse.stdout_tail }}
      Work file-by-file. Preserve all existing behaviour.
    verify_command: "npm test"
    tags: [implementation]

  - id: verify-no-regression
    depends_on: [refactor]
    command: "npm run test:integration"
    allow_failure: true
    tags: [testing]

  - id: review
    <<: *defaults
    model: sonnet
    agent: code-reviewer
    depends_on: [refactor, verify-no-regression]
    context_from: ["*"]
    context_mode: map_reduce
    prompt: |
      Review the refactoring changes.
      Verify: no behaviour changes, improved code quality, tests pass.
    judge:
      preset: code_quality
    tags: [review]
```

**Key decisions**:
- Analyse and implement are separate tasks — the AI plans before coding
- `escalation: [sonnet, opus]` auto-upgrades model on failure
- `context_mode: map_reduce` synthesises all upstream output for the reviewer
- `verify_command: "npm test"` runs after refactoring to catch regressions
- Integration tests use `allow_failure: true` — soft failure doesn't block review

**Pitfalls to avoid**:
- P27: Don't use haiku for large file analysis — it will timeout or miss changes
- P5: Set reasonable `timeout_sec` — don't retry with the same timeout
- P17: Test your verify_command manually before putting it in the plan

**Expected cost**: $3.00-6.00 for 4 tasks

---

## Recipe 3: Security Audit

Full security scan with CWE-mapped evaluation criteria.

```yaml
version: 1
name: security-audit
workspace_root: "C:/projects/my-app"
max_cost_usd: 15.0

tasks:
  - id: dependency-scan
    command: ["npm", "audit", "--json"]
    allow_failure: true
    tags: [security, scan]

  - id: code-audit
    engine: claude
    model: opus
    reasoning_effort: high
    agent: security-engineer
    depends_on: [dependency-scan]
    context_from: [dependency-scan]
    timeout_sec: 900
    prompt: |
      Perform a comprehensive security audit of the codebase.
      Focus on OWASP Top 10 vulnerabilities.
      {{ dependency-scan.stdout_tail }}
      For each finding, provide: severity, CWE ID, location, remediation.
    judge:
      preset: cwe_top_25
    tags: [security, audit]

  - id: fix-critical
    engine: claude
    model: sonnet
    depends_on: [code-audit]
    context_from: [code-audit]
    max_retries: 2
    prompt: |
      Fix all CRITICAL and HIGH severity findings from the security audit.
      {{ code-audit.stdout_tail }}
    verify_command: "npm test"
    tags: [security, fix]

  - id: verify-fixes
    engine: claude
    model: sonnet
    agent: security-engineer
    depends_on: [fix-critical]
    context_from: [code-audit, fix-critical]
    context_mode: summarized
    prompt: |
      Verify that the critical security fixes are correct and complete.
      Check that no new vulnerabilities were introduced.
    judge:
      preset: cwe_injection
      on_fail: retry
    tags: [security, verify]
```

**Key decisions**:
- Opus + high reasoning for the initial audit (security-critical)
- Sonnet for fixes (cheaper, guided by audit findings)
- `cwe_top_25` preset on audit, `cwe_injection` on verification
- Dependency scan runs first and feeds into code audit via context
- `timeout_sec: 900` for opus — security audits need time

**Pitfalls to avoid**:
- P19: Don't run too many opus tasks in parallel (rate limits, $$$)
- P4: CWE presets use `aggregation: min` — every criterion must pass
- P20: Ensure `npm` is on PATH in the subprocess environment

**Expected cost**: $8.00-15.00 for 4 tasks (opus is expensive)

---

## Recipe 4: Test Backfill

Add tests to under-tested modules. Use verify_command to ensure tests pass.

```yaml
version: 1
name: test-backfill
workspace_root: "C:/projects/my-app"
max_cost_usd: 5.0
max_parallel: 3

_test: &test_defaults
  engine: claude
  model: sonnet
  max_retries: 1
  timeout_sec: 600
  tags: [testing]

tasks:
  - id: test-auth
    <<: *test_defaults
    prompt: |
      Write pytest tests for src/auth.py.
      Cover: login, logout, token refresh, permission checks.
      Use tmp_path for file ops, monkeypatch for mocking.
    verify_command: "py -m pytest tests/test_auth.py -x -q"

  - id: test-models
    <<: *test_defaults
    prompt: |
      Write pytest tests for src/models.py.
      Cover: creation, validation, serialization, edge cases.
    verify_command: "py -m pytest tests/test_models.py -x -q"

  - id: test-api
    <<: *test_defaults
    depends_on: [test-auth, test-models]
    prompt: |
      Write integration tests for src/api.py.
      Cover: endpoints, error responses, auth middleware.
    verify_command: "py -m pytest tests/test_api.py -x -q"

  - id: coverage-check
    depends_on: [test-auth, test-models, test-api]
    command: "py -m pytest tests/ --cov=src --cov-fail-under=80 -q"
    tags: [testing, coverage]
```

**Key decisions**:
- `max_parallel: 3` — independent test modules run in parallel
- `verify_command` runs the specific test file — catches failures immediately
- Coverage check at the end ensures the backfill hit the target
- YAML anchors keep test tasks consistent

**Pitfalls to avoid**:
- P6: Test verify_command locally first — path issues are common
- P8: Use list-format commands on Windows to avoid cmd.exe issues
- P23: Be specific about what to test — "Write tests" is too vague

**Expected cost**: $2.00-4.00 for 4 tasks

### Why this shape is "ideal-fit for Maestro"

An internal backfill run was among the cleanest runs we have on record — a multi-task run that finished quickly, cheaply, with high parallelism. The shape is worth recognising:

- **Homogeneous tasks**: read one source module, write one test file, run one `pytest` invocation. Every task has the same structure.
- **No `context_from` between LLM tasks**: each task reads its own input from disk independently. No SEC016 risk, no context-budget juggling.
- **Verify is the source of truth**: `pytest test_X.py -q` is a perfect deterministic verify — exit 0 / non-zero, no "did the agent really do what we asked" ambiguity.
- **Bounded creativity**: enumerate the test cases the agent should cover, then allow extras "if obvious". You get disciplined exploration, not invention.

When you can shape the work like this, prefer it over plans with LLM-to-LLM `context_from` chains. Reserve chained context for cases where there is no alternative.

### Haiku-safe sub-shape: small modules

If the source modules are **under ~300 LOC each** and the test cases are concrete (function-level coverage of pure logic, not integration), the backfill shape is haiku-safe:

```yaml
_test: &test_defaults
  engine: claude
  model: haiku            # was: sonnet
  max_retries: 1
  timeout_sec: 600
  tags: [testing]
```

- **Sonnet baseline**: a modest per-task cost, tens of seconds wall.
- **Haiku estimate** for the same shape: substantially cheaper per task (~70-75 % cheaper), comparable wall time.
- **Quality risk**: low when the module is small and tests are concrete. The verify_command (`pytest -q`) catches any haiku regression — this is exactly what `verify_command + max_retries: 1` is designed for.

Stay on sonnet (or escalate) when:
- Modules are >300 LOC or contain non-trivial control flow
- Tests need integration setup (DB fixtures, web server, mocked external APIs)
- The prompt is open-ended ("review the module and pick the test cases yourself")

For mixed plans — small modules + one large module — set `model: haiku` per-task on the small ones and leave the large one on sonnet, rather than downshifting the whole plan.

### Signature-dump sub-shape: heavy reader tasks

When a test-backfill task needs to inspect a *large* module to confirm method signatures, the agent often re-reads the same file multiple times during exploration. An internal post-mortem caught one such task with a costly token count — sonnet cost dominated by repeated re-reads of a single large service module.

The fix is one cheap upstream haiku task that emits a 100-200 line signature dump per method, consumed by the downstream test-writer via `context_from`:

```yaml
tasks:
  # Wave 0a: cheap signature extraction (~$0.05 total, all tasks share input)
  - id: extract-signatures
    engine: claude
    model: haiku
    workdir: "{{ workspace_root }}"
    prompt: |
      Read src/services/large_service.py and emit a compact signature dump:
      for each public method, output its full signature, docstring summary,
      and the names of any external modules it imports lazily. Limit to
      ~200 lines total. No prose, no commentary.
    output_scope: [".maestro-tmp/signatures.md"]

  # Wave 0b: parallel test writers consume the dump (no re-reads of large_service.py)
  - id: test-method-a
    <<: *test_defaults
    depends_on: [extract-signatures]
    context_from: [extract-signatures]
    context_mode: raw                     # signatures are tight + structured
    context_budget_tokens: 4000
    prompt: |
      Write pytest tests for `MethodA` in src/services/large_service.py.
      The signature dump is in your context — use it to confirm argument
      names and types without re-reading the source file.
    verify_command: ["py", "-m", "pytest", "tests/test_method_a.py", "-q"]
```

This is one of the few cases where `context_from` between engine tasks is the *correct* choice — the upstream output is intentionally tight and structured (a signature dump, not freeform LLM prose). The cheap haiku extraction task costs a small fraction of what it saves on the heaviest sonnet reader, and pre-empts the cost dispersion observed across a homogeneous fan-out.

Use this sub-shape when you see one task in a homogeneous fan-out coming back ≥2× more expensive than its peers and the prompt explicitly asks the agent to "read X first to confirm the signature".

---

## Recipe 5: Bug Fix with Regression Check

Fix a specific bug and verify no regressions.

```yaml
version: 1
name: fix-login-crash
workspace_root: "C:/projects/my-app"
max_cost_usd: 3.0
fail_fast: true

tasks:
  - id: reproduce
    command: "py -m pytest tests/test_auth.py::test_login_empty_password -x"
    allow_failure: true
    tags: [debug]

  - id: fix
    engine: claude
    model: sonnet
    depends_on: [reproduce]
    context_from: [reproduce]
    max_retries: 1
    prompt: |
      Fix the crash in login when password is empty.
      {{ reproduce.stdout_tail }}
      The test test_login_empty_password should pass after the fix.
    verify_command: "py -m pytest tests/test_auth.py -x -q"
    tags: [fix]

  - id: regression
    depends_on: [fix]
    command: "py -m pytest tests/ -x -q"
    tags: [testing]
```

**Key decisions**:
- Reproduce first — captures the error output for context
- Fix task gets the error output via `context_from`
- `verify_command` runs the specific test module
- Full regression suite runs after the fix

**Expected cost**: $0.50-1.50 for 1 engine task

---

## Recipe 6: Multi-Engine Plan

Use the best engine for each task type.

```yaml
version: 1
name: multi-engine-feature
workspace_root: "C:/projects/my-app"
max_cost_usd: 10.0

defaults:
  timeout_sec: 600
  claude:
    model: sonnet
  codex:
    model: "5.4"
    reasoning_effort: medium

tasks:
  - id: design
    engine: claude
    model: opus
    reasoning_effort: high
    prompt: |
      Design the architecture for a real-time notification system.
      Output: component diagram, API contracts, data flow.
    tags: [architecture]

  - id: implement-backend
    engine: codex
    depends_on: [design]
    context_from: [design]
    prompt: |
      Implement the notification backend based on the architecture.
      {{ design.stdout_tail }}
    verify_command: "py -m pytest tests/test_notifications.py -x -q"
    fallback_engine: claude
    fallback_model: sonnet
    tags: [implementation]

  - id: implement-frontend
    engine: claude
    depends_on: [design]
    context_from: [design]
    context_mode: summarized
    prompt: |
      Implement the notification UI components.
      {{ design.summary }}
    tags: [implementation]

  - id: review
    engine: claude
    agent: code-reviewer
    depends_on: [implement-backend, implement-frontend]
    context_from: ["*"]
    context_mode: map_reduce
    prompt: |
      Review the full notification system implementation.
    judge:
      preset: code_quality
    tags: [review]
```

**Key decisions**:
- Opus for architecture (needs deep reasoning)
- Codex for backend (fast, good at implementation)
- Claude for frontend (better at UI patterns)
- `fallback_engine: claude` on Codex task — if Codex CLI is down, falls back
- `context_mode: map_reduce` synthesises all upstream for the final review

**Expected cost**: $4.00-8.00 for 4 tasks

---

## Recipe 7: Watch Mode — You Write the Strategy, Maestro Does the Work

> *"Human writes program.md (the strategy). Agent writes the code. You sleep."*
> — Inspired by [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch)

The autonomous improvement loop. You define **what** to optimise and
**how to measure** it. Maestro iterates overnight: try a change, measure
the metric, keep improvements, revert regressions. You wake up to a
better codebase with clean git history.

### The 3 things that matter

| File | Who Controls | Purpose |
|------|-------------|---------|
| **Your plan YAML** | You | The strategy — what tasks to run, what metric to chase |
| **`program.md`** | You | Research directions, constraints, priorities for the agent |
| **Your codebase** | Maestro | The agent makes changes, Maestro measures and decides |

### What you wake up to

- **Improved code** — only winning changes kept
- **`experiments.jsonl`** — every iteration logged with metric, cost, duration
- **Clean git history** — improvements committed, regressions reverted
- **Knowledge archive** — lessons extracted for future runs

### Example: Push test coverage from 60% to 90%

```yaml
version: 1
name: improve-test-coverage
workspace_root: "C:/projects/my-app"

watch:
  mode: improve
  metric: coverage_pct
  metric_direction: higher_is_better
  metric_source: stdout_regex
  metric_pattern: "^TOTAL.*?(\\d+)%"
  target_metric: 90
  max_iterations: 20
  max_total_steps: 100
  iteration_budget_sec: 300       # 5 min per iteration — keeps experiments comparable
  max_cost_usd: 25.0
  on_regression: rollback
  plateau_threshold: 3
  plateau_action: escalate_model
  program_md: program.md

tasks:
  - id: add-tests
    engine: claude
    model: sonnet
    prompt: |
      Analyse the coverage report and add tests for uncovered code.
      {{ watch.history }}
      {{ watch.program }}
      Focus on the files with lowest coverage first.
    max_retries: 1
    tags: [testing]

  - id: run-coverage
    depends_on: [add-tests]
    command: "py -m pytest tests/ --cov=src --cov-report=term -q"
    tags: [coverage]
```

And `program.md`:
```markdown
# Test Coverage Strategy

## Priority
1. Core business logic (src/models/, src/services/) — highest value
2. API endpoints (src/api/) — user-facing
3. Utilities (src/utils/) — lowest priority

## Constraints
- Do NOT mock database calls — use the test fixtures in conftest.py
- Each test file should test ONE module
- Use tmp_path for file operations, never write to the repo

## What NOT to do
- Don't test third-party libraries
- Don't write trivial getter/setter tests
- Don't duplicate existing test coverage
```

### Key decisions

| Decision | Why |
|----------|-----|
| `iteration_budget_sec: 300` | Fixed wall-clock per iteration — every experiment is comparable |
| `on_regression: rollback` | Bad changes are reverted instantly via `git reset` |
| `plateau_threshold: 3` | After 3 stale iterations, escalate to a stronger model |
| `max_total_steps: 100` | Hard cap prevents runaway loops (even past plateau) |
| `max_cost_usd: 25.0` | Budget ceiling — stops when money runs out |
| `program_md` | Your strategy doc — the agent reads this every iteration |
| `{{ watch.history }}` | Agent sees what worked and what didn't in prior iterations |
| `target_metric: 90` | Stops when coverage reaches 90% — no wasted iterations |

### Real-world results

From Maestro self-development (improving its own test suite):

| Metric | Before | After | Iterations | Cost |
|--------|--------|-------|-----------|------|
| Test count | 6726 | 10570 | ~15 sessions + 1 overnight | ~$66 total |
| Module coverage grade | 5 × C, 5 × B | 19 × A, 3 × B | — | — |
| Watch overnight run | 9782 | 10568 | 13 iterations, 1h46m | $26.18 |

### Other watch mode use cases

**Reduce bundle size**:
```yaml
watch:
  metric: bundle_kb
  metric_direction: lower_is_better
  metric_source: stdout_regex
  metric_pattern: "Total: (\\d+\\.?\\d*) KB"
  target_metric: 200
```

**Improve benchmark performance**:
```yaml
watch:
  metric: ops_per_sec
  metric_direction: higher_is_better
  metric_source: stdout_regex
  metric_pattern: "(\\d+) ops/sec"
```

**Reduce lint warnings**:
```yaml
watch:
  metric: warning_count
  metric_direction: lower_is_better
  metric_source: stdout_regex
  metric_pattern: "(\\d+) warnings?"
  target_metric: 0
```

**Expected cost**: $5.00-25.00 depending on iterations and target difficulty

---

## Anti-Patterns

### Don't: All-Opus Plans
```yaml
# BAD: $15-25 per run
tasks:
  - id: fix-typo
    engine: claude
    model: opus  # Opus for a typo fix? $3+ wasted
```
**Fix**: Use haiku for trivial tasks, sonnet for standard work.

### Don't: Missing Verify Commands
```yaml
# BAD: No way to know if the task actually worked
tasks:
  - id: implement
    engine: claude
    prompt: "Add the feature"
    # No verify_command — success is just "exit 0"
```
**Fix**: Always add `verify_command` on implementation tasks.

### Don't: Blanket allow_failure
```yaml
# BAD: Defeats quality gates
tasks:
  - id: implement
    allow_failure: true  # Why? Now failures are invisible
```
**Fix**: Only use `allow_failure` on non-critical scans and optional checks.

### Don't: Vague Prompts
```yaml
# BAD: AI will ask clarifying questions instead of working
prompt: "Improve the code"
```
**Fix**: Be specific — what files, what changes, what the output should look like.

### Don't: Generic Persona Prompts
```yaml
# BAD: Generic roleplay hurts factual accuracy on code tasks
append_system_prompt: "You are a senior full-stack developer with 15 years of experience"
```
**Fix**: Instead of generic personas, provide **specific constraints and context**:
architecture rules, file patterns, coding standards, review criteria. Research
shows generic role prompts reduce accuracy by ~3.6pp on coding tasks because they
activate instruction-following at the expense of factual recall (USC, 2026).
Use `agent:` references that define concrete responsibilities, not identities.
See [the study](https://arxiv.org/html/2603.18507v1) for details.

### Don't: String Commands on Windows
```yaml
# BAD: shell=True uses cmd.exe on Windows
command: "bash -c 'echo hello'"
```
**Fix**: Use list-format: `["C:/Program Files/Git/bin/bash.exe", "-c", "echo hello"]`

### Don't: Assume the Engine Knows Your Language Version
```yaml
# BAD: Engine defaults to latest PHP/Python/Node
prompt: "Add constants to the service class"
# Claude generates PHP 8.3 typed constants in a PHP 8.2 project → parse error
```
**Fix**: Always specify the version: `"Use PHP 8.2 syntax only. No typed constants."` and set `max_retries: 1` so verify_command errors trigger a self-correcting retry.

### Don't: Use `file_contains` for Semantic Checks
```yaml
# BAD: Checks for literal "is_internal" but controller delegates to repository
assert:
  - type: file_contains
    path: src/Controllers/TicketController.php
    pattern: "is_internal"
```
**Fix**: Use `file_contains` only for structural checks (class names, `extends`, `use`). For semantic validation, use `verify_command` with a script or `llm-rubric` judge criteria.

---

## Cost Optimisation Checklist

0. **Run `maestro check <plan.yaml>` first**: validate + audit in one pass, surfaces W20 (retry escape valves), SEC001 (no budget), and audit findings before you spend a cent on the run
1. **Start cheap**: haiku/codex@low for first attempt
2. **Escalate on failure**: `escalation: [haiku, sonnet, opus]` (or omit if `verify_command` already gives the retry feedback to differ)
3. **Set budget**: `max_cost_usd` on every plan (`scaffold --strict-defaults` does this for you)
4. **Use verify_command**: Zero-cost validation; with `max_retries: 1` it's the cheapest W20 escape valve
5. **Use guard_command**: Zero-cost output validation via stdin pipe
6. **Cache results**: Don't re-run successful tasks (`cache: true` is default); aliases and arg order are normalised pre-hash, so cosmetic differences hit the cache
7. **Tag for reruns**: `--tags fix` only re-runs tagged tasks
8. **Use context_mode**: `layered` saves 40-65% tokens vs `raw`; RRF fusion picks the most relevant upstreams automatically
9. **Set timeout_sec**: Prevent tasks from running forever
10. **Use ollama locally**: $0 cost for validation and simple checks
11. **Use `--set` for variants**: `--set env=prod` lets you re-run plans across environments without editing YAML; cache keys vary per `--set` value

---

## Verification Stack (Zero to Expensive)

Layer your verification from cheapest to most expensive:

```
1. verify_command  — shell command, $0        (syntax check, test runner)
2. guard_command   — stdin pipe, $0           (output format validation)
3. assert          — deterministic, $0        (contains, regex, is-json)
4. judge criteria  — LLM call, $0.05-0.50    (quality assessment)
5. judge rubric    — LLM + scoring, $0.10-1.00 (structured evaluation)
6. judge g_eval    — two-phase LLM, $0.20-2.00 (highest consistency)
```

**Rule**: Use the cheapest layer that catches the issue. Don't pay for
LLM judge when `verify_command: "npm test"` would catch it.

---

## Recipe 8: Multi-File TypeScript Codegen

Generate a full subsystem (DB migration, service, controller, UI page) across
multiple tasks. The critical lesson: **verify integration, not just file existence.**

Based on a real post-mortem: every task passed, but several critical integration
bugs made the system non-functional. Every bug was invisible to per-file keyword checks.

```yaml
version: 1
name: arena-subsystem
workspace_root: "C:/projects/my-app"
max_cost_usd: 12.0
fail_fast: true

_impl: &impl
  engine: claude
  model: sonnet
  max_retries: 1
  timeout_sec: 900
  edit_policy: efficient

tasks:
  # Wave 1: Schema + Protocol (the contract layer)
  - id: create-migration
    <<: *impl
    prompt: |
      Create the SQL migration for the arena system.
      Tables: tournaments, tournament_agents, rounds, trades.
      Include: id, created_at, updated_at on every table.
    verify_command: ["py", "-c", "content = open('prisma/migrations/arena.sql').read(); assert 'CREATE TABLE' in content; print('OK')"]
    tags: [schema]

  - id: create-protocol
    <<: *impl
    prompt: |
      Create apps/api/src/arena/arena.protocol.ts with:
      - TournamentConfig interface (name, startDate, endDate, initialCapital)
      - TournamentResponse: { tournament: TournamentRow }
      - TournamentsListResponse: { tournaments: TournamentRow[] }
      - ScoreboardResponse: { standings: ScoreboardEntry[] }
      - AgentDecision interface (action, ticker, quantity, reasoning)
      Export ALL types. This is the single source of truth for API shapes.
    verify_command: "npx tsc --noEmit --project apps/api/tsconfig.json"
    tags: [protocol]

  # Wave 2: Service + Agents (consume protocol)
  - id: create-service
    <<: *impl
    depends_on: [create-migration, create-protocol]
    context_from: [create-migration, create-protocol]
    context_mode: raw                # CRITICAL: raw preserves exact type definitions
    context_budget_tokens: 12000     # generous — interface fidelity matters
    prompt: |
      Create arena.service.ts. Import ALL types from arena.protocol.ts.
      Return types MUST match the Response interfaces exactly.
      Do NOT access the database from anywhere except this service.
    verify_command: "npx tsc --noEmit --project apps/api/tsconfig.json"
    tags: [implementation]

  - id: create-agents
    <<: *impl
    depends_on: [create-protocol]
    context_from: [create-protocol]
    context_mode: raw
    prompt: |
      Create agent classes implementing BaseAgent.
      Constructor signature: (agentId: string) — all agents MUST match.
      Import AgentDecision from arena.protocol.ts.
    verify_command: "npx tsc --noEmit --project apps/api/tsconfig.json"
    tags: [implementation]

  # Wave 3: Controller (consumes service + protocol)
  - id: create-controller
    <<: *impl
    depends_on: [create-service, create-protocol]
    context_from: [create-service, create-protocol]
    context_mode: raw
    context_budget_tokens: 12000
    prompt: |
      Create arena.controller.ts.
      - Inject ArenaService. Delegate ALL operations to it.
      - Do NOT access the database directly (no this.db, no raw queries).
      - Response shapes MUST match arena.protocol.ts types exactly.
      - POST endpoints MUST accept a JSON body (Fastify rejects empty POSTs).
    verify_command: "npx tsc --noEmit --project apps/api/tsconfig.json"
    # guard_command receives stdout via stdin. List-form keeps the YAML valid;
    # one-liner avoids the block-scalar-inside-flow-sequence parse error that
    # the multi-line variant of this snippet used to have.
    guard_command:
      - py
      - -c
      - "import sys; c = sys.stdin.read(); assert '.db.' not in c, 'Controller must not access DB directly — delegate to service'; print('OK')"
    tags: [implementation]

  # Wave 4: Dashboard (consumes protocol)
  - id: create-dashboard
    <<: *impl
    depends_on: [create-controller, create-protocol]
    context_from: [create-protocol]       # Protocol, NOT controller output
    context_mode: raw                     # Exact types, not summaries
    prompt: |
      Create the arena dashboard page.
      Import types from arena.protocol.ts for ALL API responses.
      API response shapes are defined in the protocol — use them exactly.
      POST requests MUST include a JSON body (Content-Type: application/json).
    verify_command: "npx tsc --noEmit"
    tags: [frontend]

  # Wave 5: Integration check + Review
  - id: integration-check
    depends_on: [create-controller, create-dashboard, create-agents]
    command: "npx tsc --noEmit --project apps/api/tsconfig.json"
    description: "Full TypeScript compilation — catches cross-file type mismatches"
    tags: [verify]

  - id: review
    engine: claude
    model: sonnet
    agent: code-reviewer
    depends_on: [integration-check]
    context_from: ["*"]
    context_mode: map_reduce
    cache: false
    prompt: |
      Review the entire arena subsystem. Check:
      1. Controller delegates to service (no direct DB access)
      2. All API response shapes match protocol types
      3. Dashboard fetch types match protocol types
      4. All agent classes have identical constructor signatures
      5. POST endpoints receive JSON bodies
    judge:
      preset: code_quality
      on_fail: fail           # NOT warn — broken code must not pass
    tags: [review]
```

**Key decisions**:
- **Protocol-first architecture** — a shared types file is the single source of truth.
  Every consumer imports from it. This prevents the #1 failure mode: producer and
  consumer disagreeing on response shapes.
- **`context_mode: raw` for interface tasks** — `layered` or `summarized` may
  compress away the exact type definitions that consumers need. When interface
  fidelity matters, pay the token cost for full context.
- **`tsc --noEmit` as verify_command** — catches cross-file type mismatches at
  zero cost. This single command would have caught ~4/9 critical bugs in the
  post-mortem.
- **`guard_command` for architectural rules** — "controller must not access DB
  directly" is a grep-level check, not an LLM task.
- **`on_fail: fail` on the review** — if the reviewer says the code is broken,
  the run must fail. `on_fail: warn` means "I don't care about the review result."

**Pitfalls to avoid**:
- P28: Don't rely on per-file keyword checks (`assert 'className' in content`).
  They miss every integration bug.
- P29: Don't use `on_fail: warn` on review tasks when correctness matters.
- P5: Set generous timeouts (900s) for multi-file codegen tasks.
- Always include the migration in controller/service context — prevents schema
  mismatch bugs.

**What this pattern catches that single-file checks miss**:

| Bug Class | How This Recipe Catches It |
|-----------|--------------------------|
| Response type mismatch | `tsc --noEmit` (protocol is shared import) |
| Missing DB column | `tsc --noEmit` (if types are wired to schema) |
| DI bypass (direct DB access) | `guard_command` grep check |
| Constructor signature mismatch | `tsc --noEmit` (shared interface) |
| Form payload mismatch | `tsc --noEmit` (protocol types on both sides) |
| Empty POST body | Explicit prompt instruction + review |

**Expected cost**: $6.00-12.00 for 8 tasks

---

## Recipe 9: Deep Codebase Analysis with MCP Tools

Use external knowledge graph tools via MCP to give tasks deep structural
awareness of your codebase — call chains, blast radius, functional clusters —
without building that analysis into the plan itself.

Several tools expose codebase knowledge graphs via MCP. Pick the one that
fits your licence and stack:

| Tool | Licence | Language | MCP Tools | Best For |
|------|---------|----------|-----------|----------|
| [code-review-graph](https://github.com/tirth8205/code-review-graph) | **MIT** | Python | 22 | Claude Code / Python shops, blast radius (8.2× fewer tokens) |
| [codegraph](https://github.com/optave/codegraph) | **Apache-2.0** | TypeScript | 30+ | Richest MCP surface, CI gates, dataflow analysis |
| [Codemem](https://github.com/cogniplex/codemem) | **Apache-2.0** | Rust | 32 | Temporal memory, SCIP cross-refs, graph-vector hybrid |
| [GitNexus](https://github.com/abhigyanpatwari/GitNexus) | PolyForm NC | TypeScript | 7 | Community detection, 14-lang AST, process tracing |

GitNexus is **PolyForm Noncommercial** — commercial use requires an enterprise
licence. The other three are fully permissive. The MCP pattern itself is
tool-agnostic — the YAML below works with any of them (adjust `command` and
tool names).

**When to use this**: Large codebases (10K+ files) where `context_mode: structural`
or `knowledge_graph` isn't enough. The external tool indexes the repo once; every
task query is instant.

**Prerequisites**: Install and index your repo with your chosen tool:
```bash
# code-review-graph (Python, MIT) — recommended for commercial use
pip install code-review-graph
code-review-graph build

# codegraph (TypeScript, Apache-2.0)
npm install -g @optave/codegraph
codegraph build

# GitNexus (TypeScript, PolyForm NC — non-commercial only)
npm install -g gitnexus
gitnexus index .
```

```yaml
version: 1
name: impact-aware-refactor
workspace_root: "C:/projects/my-app"
max_cost_usd: 10.0
fail_fast: true

mcp_servers:
  - name: gitnexus
    transport: stdio
    command: ["gitnexus", "mcp"]

_impl: &impl
  engine: claude
  model: sonnet
  max_retries: 1
  timeout_sec: 900

tasks:
  - id: analyse-impact
    <<: *impl
    mcp_tools: [gitnexus]
    prompt: |
      Use the gitnexus impact tool to analyse the blast radius of
      changes to src/scheduler.py.
      List all functions that call into the scheduler, grouped by
      functional cluster. Identify the 3 highest-risk downstream consumers.
    tags: [analysis]

  - id: refactor
    <<: *impl
    depends_on: [analyse-impact]
    context_from: [analyse-impact]
    context_mode: raw
    mcp_tools: [gitnexus]
    prompt: |
      Refactor the scheduler's _select_tasks() function.
      {{ analyse-impact.stdout_tail }}
      Use gitnexus context tool to check each downstream consumer
      before modifying shared interfaces.
    verify_command: "py -m pytest tests/test_scheduler.py -x -q"
    tags: [implementation]

  - id: verify-downstream
    <<: *impl
    depends_on: [refactor]
    context_from: [analyse-impact, refactor]
    context_mode: summarized
    mcp_tools: [gitnexus]
    prompt: |
      Use gitnexus detect_changes to map the refactored diff to
      affected symbols and processes. Verify that no downstream
      consumer is broken by the interface changes.
    judge:
      preset: code_quality
    tags: [verify]
```

If the MCP provider writes shared state outside each task's worktree, add
`is_concurrency_safe: false` on that `mcp_servers` entry. Maestro will then
serialize parallel `worktree: true` tasks that share the provider, which is
safer for local indexes, mutable caches, or single-writer daemons.

**Adapting for other MCP tools**:

```yaml
# code-review-graph (Python, MIT)
mcp_servers:
  - name: crg
    transport: stdio
    command: ["code-review-graph", "mcp"]
# Tools: get_impact_radius_tool, get_review_context_tool, query_graph_tool,
#   detect_changes_tool, list_flows_tool, get_affected_flows_tool

# codegraph (TypeScript, Apache-2.0)
mcp_servers:
  - name: codegraph
    transport: stdio
    command: ["codegraph", "mcp"]
# Tools: fn_impact, diff_impact, query, context, semantic_search, hotspots
```

**Key decisions**:
- `mcp_tools: [gitnexus]` gives specific tasks access to the knowledge graph
- Impact analysis runs first — the refactor task knows exactly what it might break
- `detect_changes` in the verification step maps the actual diff to affected code
- `context_mode: raw` on the refactor task — impact data needs full fidelity
- Tests via `verify_command` catch regressions at zero cost

**When NOT to use this**:
- Small repos (<50 files) — `context_mode: structural` is enough
- No persistent index — if you can't pre-index, the MCP overhead isn't worth it
- Budget-sensitive runs — MCP tool calls add latency and token cost

**Complementary context modes**:
- Use `context_mode: structural` (built-in, zero-deps) for simpler blast radius analysis
- Use `context_mode: knowledge_graph` (built-in) for entity/relationship extraction from task output
- Use MCP tools (this recipe) for deep cross-file analysis with call chains and community clusters

**Expected cost**: $3.00-8.00 for 3 tasks (plus MCP tool call tokens)

---

## Recipe 10: Structured Output Validation (output_schema)

Replace the brittle "ask the agent for JSON, parse stdout_tail with regex"
pattern with a typed schema. Downstream tasks consume validated fields via
`{{ task-id.output.field }}` template variables — no parsing, no
hallucinated keys.

```yaml
version: 1
name: triage-and-fix
workspace_root: "C:/projects/my-app"
max_cost_usd: 4.0

_engine: &engine
  engine: claude
  model: sonnet
  max_retries: 1

tasks:
  # Step 1: Triage produces a typed verdict
  - id: triage
    <<: *engine
    prompt: |
      Read the failing pytest output below and decide:
      - severity: one of "critical" | "major" | "minor"
      - root_cause_file: the source file most likely responsible
      - estimated_loc: integer line-of-code estimate for the fix

      Emit your decision as a JSON object matching the schema. Output ONLY
      the JSON, no commentary.

      ```
      {{ failure_log }}
      ```
    output_schema:
      type: object
      required: [severity, root_cause_file, estimated_loc]
      properties:
        severity:
          type: string
          enum: ["critical", "major", "minor"]
        root_cause_file:
          type: string
          minLength: 1
        estimated_loc:
          type: integer

  # Step 2: Fix consumes validated fields, no parsing required
  - id: fix
    <<: *engine
    depends_on: [triage]
    prompt: |
      Severity: {{ triage.output.severity }}
      Root cause file: {{ triage.output.root_cause_file }}
      Estimated LOC: {{ triage.output.estimated_loc }}

      Apply a focused fix to {{ triage.output.root_cause_file }}. Stay within
      the LOC budget unless the fix genuinely requires more.
    verify_command: ["py", "-m", "pytest", "-x", "-q"]
```

**Key decisions**:
- `output_schema` validates after a successful run (parses stdout_tail as JSON, then markdown code block, then first `{...}` block). Failure marks the task as failed and counts toward `max_retries`.
- `{{ triage.output.severity }}` resolves to the validated string at runtime. Nested objects and arrays are JSON-encoded; primitives are passed as-is.
- Schema is a JSON Schema subset: `type`, `properties`, `required`, `items`, `enum`, `minLength`, `maxLength` are supported (recursive depth limit 20).
- Use `schema_file:` instead of inline `schema:` when sharing a schema across multiple tasks.

**Pitfalls to avoid**:
- Don't combine with `judge: contains` on the same task — judges check `stdout_tail` text, but `output_schema` already validates structure. Pick one.
- Don't ask the agent for free-form JSON in the prompt. Always specify the exact field names and types in the prompt body so the agent doesn't invent keys.
- Schema validation runs on the engine's stdout, not on files written. For "did this task write file X" assertions, use `assert: file_contains` instead.

**Expected cost**: $0.20-1.00 for the triage + fix pair (sonnet baseline). Adds zero overhead vs. raw JSON prompting.

---

## Recipe 11: Batched Item Processing (batch + max_per_call)

When you need to process N similar items (files, rows, modules, URLs) and a
matrix expansion would create too many tasks, use `batch:` to chunk them into
a single task that runs the agent multiple times within one task lifecycle:

```yaml
version: 1
name: refactor-deprecated-imports
workspace_root: "C:/projects/my-app"
max_cost_usd: 6.0

tasks:
  - id: replace-imports
    engine: claude
    model: sonnet
    batch:
      items:
        - "src/a/legacy_v1.py"
        - "src/b/legacy_v1.py"
        - "src/c/legacy_v1.py"
        - "src/d/legacy_v1.py"
        - "src/e/legacy_v1.py"
        - "src/f/legacy_v1.py"
      max_per_call: 2          # 2 items per agent invocation = 3 chunks
      template: |
        Update `{{ batch.item }}`:
        replace `from legacy_v1 import X` with `from legacy_v2 import X`.
        If a symbol was renamed in v2, use the new name.
        Run the file's local tests after editing to confirm the change works.
    prompt: |
      You will edit a small batch of files in this invocation. The list of
      files is provided per item via the template below; process each one
      and emit a short summary of what changed.
    verify_command: ["py", "-m", "pytest", "-q"]
    tags: [refactor, batch]
```

**Key decisions**:
- `batch.items` is the static list to chunk (use `matrix:` if items expand combinatorially across keys).
- `batch.max_per_call` controls chunk size; larger values reduce overhead but blow the agent's context if items are large. Default is 1 (one item per call); 2-5 is the typical sweet spot for small files.
- `batch.template` MUST contain `{{ batch.item }}` (E057 if missing). The template is rendered per-item inside each chunk's prompt.
- Each chunk emits a `batch_chunk_complete` event with `task_id`, `chunk`, `total_chunks`, `items_in_chunk`, `exit_code`.
- `verify_command` runs ONCE after all chunks finish, against the final state of the workspace. For per-chunk verification, set `guard_command` instead.

**Pitfalls to avoid**:
- E057 if `template` is missing or doesn't contain `{{ batch.item }}`.
- E058 if `max_per_call < 1`. E060/E062 if `batch:` is set on a command/group task or combined with `matrix:`.
- Don't mix items of wildly different sizes in one batch — the agent's attention degrades. Sort by approximate size or split into multiple `batch:` tasks.

**Expected cost**: scales linearly with item count. For 6 small files at 2 per call (3 chunks, sonnet): ~$0.40-0.80. Compare to a 6-task matrix expansion: ~$0.60-1.20 (matrix has per-task overhead). Use batch when the work is structurally identical across items.

---

## Recipe 12: Multi-Model Council Deliberation

For decisions where one model isn't reliable enough on its own — security
threat-modelling, ambiguous bug triage, architecture trade-offs — run a
council of N models that deliberate over R rounds before producing a
consolidated answer. The council pattern is `context_mode: council` plus a
`council:` block on the task.

### Star topology (default — all participants see each other every round)

```yaml
- id: triage-architecture
  engine: claude
  model: sonnet
  context_mode: council
  council:
    topology: star
    rounds: 2
    consensus_threshold: 0.7
    participants:
      - engine: claude
        model: sonnet
        role: pragmatist
      - engine: codex
        model: "5.4"
        role: theoretician
      - engine: claude
        model: sonnet
        role: safety
  prompt: |
    Decide whether to migrate the scheduler from threads to asyncio. Each
    participant offers a position, reads peers' positions in round 2, then
    a haiku consolidator emits the final recommendation.
```

### Chain topology (single-pass pipeline, each participant sees only the previous)

Useful for "draft → critique → polish" patterns. `rounds > 1` is wasted
under chain (W28); each round just re-pipelines the same chain.

```yaml
council:
  topology: chain
  rounds: 1
  participants:
    - { engine: claude, model: haiku,  role: drafter }
    - { engine: claude, model: sonnet, role: critic }
    - { engine: claude, model: opus,   role: polisher }
```

### Graph topology (custom visibility — `connections` map required)

Each participant sees only the participants listed in their `connections`
entry. Models a structured discussion (e.g. theoretician informs both
pragmatist and safety, but pragmatist and safety don't see each other).

```yaml
council:
  topology: graph
  rounds: 2
  connections:
    pragmatist: [theoretician]
    safety: [theoretician]
    theoretician: [pragmatist, safety]
  participants:
    - { engine: claude, model: sonnet, role: pragmatist }
    - { engine: codex,  model: "5.4",  role: theoretician }
    - { engine: claude, model: sonnet, role: safety }
```

E072 fires if `topology: graph` is set without a complete `connections` map.

**Key decisions**:
- Star is the safest default. Chain is the cheapest. Graph is for cases where you genuinely want to model a discussion topology.
- `rounds: 2` is the sweet spot for `star` — round 1 collects positions, round 2 lets participants react. Beyond that, returns diminish.
- `consensus_threshold` (0.0-1.0) gates the consolidation: above threshold the haiku consolidator agrees with the modal position; below it produces a "unresolved" verdict that you can act on.
- Cost: `(participants × rounds + 1) × per-call cost` for `star` / `graph`. `chain` skips the +1 consolidation. A 3-participant 2-round star at sonnet runs ~$0.30-0.60.

**Pitfalls to avoid**:
- W28 if `rounds > 1` and `topology: chain` (no benefit from extra rounds).
- W28 if `connections` is set on `star` / `chain` topologies (ignored — only `graph` uses it).
- Don't council-deliberate on tasks with deterministic verify (`pytest -q`) — a single sonnet attempt + verify_command retry is cheaper and equally reliable. Council is for *judgement* tasks where there is no oracle.

**Expected cost**: $0.30-1.50 per task depending on participants × rounds × model tiers.
