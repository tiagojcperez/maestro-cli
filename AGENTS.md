# Maestro CLI — Agent Catalog

Role definitions for the `.claude/agents/` team. Each agent has a specific expertise area, model preference, and collaboration pattern.

For the full operations manual (decision trees, pitfalls, checklists), see [docs/AGENT_OPS.md](docs/AGENT_OPS.md).

## Activation Policy

- Treat agent names as routing handles, not always-on expert personas.
- Use agents selectively for alignment-heavy work such as review, QA, safety, and plan synthesis.
- For raw factual retrieval, schema lookup, command recall, log reading, or exact contract checks, prefer source-grounded prompts without `agent:`.
- Keep prompts constraint-first: files, invariants, and acceptance criteria before any role framing.
- For mixed tasks, use a PRISM-style split: precision-first analysis, then specialist execution/review, then deterministic verification.
- See [.claude/rules/agent-routing.md](.claude/rules/agent-routing.md) and the PRISM paper: <https://arxiv.org/html/2603.18507v1>.

---

## Team Overview

| Agent | Model | Expertise | Primary Files |
|-------|-------|-----------|---------------|
| **architect** | sonnet | System design, DAG patterns, schema evolution | `models.py`, `scheduler.py`, `loader.py` |
| **python-developer** | sonnet | Implementation, module boundaries, type safety | All `src/maestro_cli/` modules |
| **cli-engineer** | sonnet | CLI contract, argparse, cross-platform shell | `cli.py`, `README.md` |
| **qa-engineer** | sonnet | pytest, fixtures, subprocess mocking, edge cases | `tests/` |
| **code-reviewer** | sonnet | Correctness, type safety, concurrency, security | All source files |
| **quality-gatekeeper** | sonnet | Output validation, regression risk, acceptance criteria | Run artifacts |
| **cost-optimizer** | sonnet | Model routing, token budget, reasoning effort tuning | Plan YAML files |
| **plan-author** | sonnet | YAML plans, DAG design, prompt sources, cost-aware shaping | Plan YAML files |
| **security-engineer** | **opus** | Subprocess security, secret handling, plugin trust | `runners.py`, `plugins.py`, `audit.py` |
| **tui-engineer** | sonnet | Textual widgets, threading, keyboard handling, CSS | `tui/` package |
| **dynamic-planner** | sonnet | Runtime sub-plan generation for `dynamic_group` tasks, allowlist-safe JSON output | `dynamic.py`, plan YAML |

---

## Role Details

### Architect

**When to use**: Feature design, schema extensions, cross-module impact assessment, data flow review.

**Escalation to opus**: Security-critical architecture changes, concurrency model rewrites, deep incident forensics.

**Key principles**: Minimal dependencies (stdlib first), functional style (module-level functions), stateless execution, fail-fast validation, engine-agnostic scheduler.

**Collaborates with**: python-developer (implementation), cli-engineer (CLI contract), plan-author (schema evolution), cost-optimizer (cost-aware design).

---

### Python Developer

**When to use**: Feature implementation, bug fixes, module changes, backward compatibility.

**Key rules**: Changes in the correct module with minimal surface area. Preserve type safety. Keep runtime, loader, cache, doctor, and docs aligned when behavior changes. Avoid coupling to specific engines.

**Collaborates with**: architect (design), qa-engineer (test coverage), code-reviewer (review), cli-engineer (CLI wiring).

---

### CLI Engineer

**When to use**: Adding subcommands, modifying flags, help text, exit codes, cross-platform shell support.

**Key rules**: Keep help text concise. Windows PowerShell and Unix shell equally first-class. CLI contract in `cli.py` must stay aligned with `README.md` and `CLAUDE.md`. New flags need `_build_parser()` + handler wiring.

**Collaborates with**: architect (interface design), python-developer (implementation), plan-author (plan contract gaps).

---

### QA Engineer

**When to use**: Writing tests, creating fixtures, achieving edge case coverage, test infrastructure.

**Key rules**: pytest only (not unittest). `tmp_path` for all file ops. ALWAYS mock `subprocess.run` for engine tests — never invoke real CLIs. Test both success AND failure paths. `@pytest.mark.parametrize` for variations.

**Collaborates with**: python-developer (test coverage), code-reviewer (review), quality-gatekeeper (acceptance).

---

### Code Reviewer

**When to use**: Reviewing PRs, code changes, architectural consistency checks.

**Escalation to opus**: Security-sensitive changes, concurrency-critical scheduler changes, large cross-cutting refactors with prior inconclusive reviews.

**Focus areas**: Type safety (PEP 604 unions, Literal types), subprocess/path/env safety, error handling patterns, cross-platform compatibility, backward compatibility for v1.x.

**Collaborates with**: architect (consistency), security-engineer (security review), quality-gatekeeper (acceptance).

---

### Quality Gatekeeper

**When to use**: Validating implementation output against acceptance criteria, pre-merge checks.

**Key rules**: Require test evidence for changed behavior. Flag regressions, security risks, missing edge cases. Check type/style consistency. Verify observability (events, logs, manifests updated).

**Collaborates with**: code-reviewer (review), qa-engineer (test evidence), cost-optimizer (cost preservation).

---

### Cost Optimizer

**When to use**: Plan cost analysis, model routing decisions, budget strategy.

**Key rules**: Route tasks to cheapest viable model/effort. Target < $1.00/task average. Enforce prompt/context minimization. Quality gates required when using lower-cost models. Escalate model tier only when conditions apply (security-critical, repeated failures, concurrency invariants).

**Decision tree**: See `docs/AGENT_OPS.md` section 3.

**Collaborates with**: plan-author (plan design), quality-gatekeeper (quality preservation), architect (cost-aware architecture).

---

### Plan Author

**When to use**: Writing YAML plans, debugging failed runs, optimizing plan structure.

**Key rules**: Plans must validate and run. Maximize useful parallelism. Choose cheapest viable engine/model. Keep prompts Windows-safe. Deterministic shell verification before expensive LLM gates.

**Pre-flight checklist**: See `docs/AGENT_OPS.md` section 11.

**Collaborates with**: architect (schema evolution), cost-optimizer (budget pressure), cli-engineer (CLI contract gaps).

---

### Security Engineer

**When to use**: Security review of code changes, secret handling, plugin trust boundaries, subprocess safety.

**Model**: opus (default) — deeper reasoning for trust boundary analysis.

**Focus areas**: Subprocess injection, path traversal, env-var leakage, filesystem access patterns, secret masking and artifact-leak posture, plugin trust boundaries and entry-point loading, CI/security-process changes (without overstating them as frozen `1.x` contract).

**Key files**: `runners.py`, `doctor.py`, `plugins.py`, `ci.py`, `ci_github_actions.py`, `ci_gitlab_ci.py`, `docs/SECURITY.md`, `docs/SECURITY_BASELINE.md`.

---

### TUI Engineer

**When to use**: TUI widget development, keyboard handling, threading issues, CSS styling.

**Escalation to opus**: Thread deadlocks, Textual framework bugs, complex CSS layout issues.

**Key rules**: Executor thread vs Textual main thread separation. `call_from_thread()` for cross-thread communication. Debounce rapid key events. Never block the main thread.

**Key files**: `tui/app.py`, `tui/widgets.py`, `tui/app.tcss`.

---

### Dynamic Planner

**When to use**: Runtime sub-plan generation for `dynamic_group: true` tasks — analyzing a codebase or problem space and producing a structured JSON plan that Maestro executes as a nested DAG.

**Key rules**: Output **valid JSON** matching the task's `output_schema` (a `tasks` array). Output is **untrusted by design** — the runtime applies a strict 7-field allowlist and `validate_plan()` before execution. Minimize task count (subprocess overhead). Use the cheapest viable model. Design for parallelism. Write self-contained, actionable prompts with exact file paths. Sequence reviews after implementation. Avoid persona inflation in generated prompts.

**Key files**: `dynamic.py`, plan YAML.

**Collaborates with**: plan-author (plan design patterns), cost-optimizer (model selection for generated tasks), security-engineer (trust boundary enforcement).

---

## Collaboration Map

```
                    ┌─────────────┐
                    │  architect  │
                    └──────┬──────┘
                           │ design
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼───┐  ┌─────▼─────┐  ┌──▼───────────┐
     │cli-engineer│  │  python-   │  │ plan-author  │
     └────────────┘  │ developer  │  └──────┬───────┘
                     └─────┬──────┘         │
                           │ impl           │ plan design
              ┌────────────┼────────┐       │
              │            │        │       │
     ┌────────▼───┐  ┌─────▼─────┐  ┌──────▼───────┐
     │qa-engineer │  │   code-   │  │cost-optimizer│
     └────────────┘  │ reviewer  │  └──────────────┘
                     └─────┬──────┘
                           │ review
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼────────┐  ┌▼───────────┐
     │quality-gatekeeper│  │  security- │
     └─────────────────┘  │  engineer  │
                          └────────────┘

     ┌──────────────┐
     │ tui-engineer │  (independent, UI-focused)
     └──────────────┘

     ┌────────────────┐
     │dynamic-planner │  (independent, runtime sub-plan generation)
     └────────────────┘
```

---

## Model Selection Policy

- **sonnet** for 90% of work — structured implementation, testing, planning, review, runtime sub-plan generation (dynamic-planner)
- **opus** only when:
  - Security-critical changes across multiple modules
  - Concurrency model rewrites (scheduler invariants)
  - Deep incident forensics where sonnet was inconclusive
  - All security review (security-engineer baseline)
  - Textual framework internal bugs (tui-engineer escalation)
