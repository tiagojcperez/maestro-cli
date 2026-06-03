# Threat Model

This document describes the security threat model for Maestro CLI: the trust
boundaries it assumes, the threats it defends against, and the concrete
mitigations it ships mapped to those threats.

It is an architectural reference, not a hardening checklist. For day-to-day
operational hardening (secrets handling, plugin trust, CI hardening,
real-engine test opt-in), see
[SECURITY_BASELINE.md](SECURITY_BASELINE.md). To **report a vulnerability**,
see the root [SECURITY.md](../SECURITY.md) -- do not open a public issue.

## What Maestro Is

Maestro is a local-first orchestrator. It schedules tasks as a DAG and runs
them by shelling out to AI engine CLIs (`codex`, `claude`, `gemini`,
`copilot`, `qwen`, `ollama`, `llama-cli`) or to raw shell commands. It runs on
the operator's own machine, with the operator's own credentials, against the
operator's own provider accounts.

This shapes the entire threat model. The interesting security surface is **not**
"can a remote attacker compromise a Maestro server" -- there is no server. It is
**"can untrusted data that flows through a run escape its intended trust
boundary"** -- for example, can model output from one task inject instructions
into the next, can poisoned cross-run memory steer future runs, or can a
declared secret leak into a log.

## Trust Boundaries

Maestro treats the following as **trusted** (operator-controlled code and
configuration):

- The authored plan YAML and any imported sub-plans.
- The engine CLIs on `PATH` and their model providers.
- Installed Python packages, including custom engine plugins discovered via the
  `maestro_cli.engines` entry-point group.
- The operator's environment and credentials.

Maestro treats the following as **untrusted** (data that may be adversarial):

| Boundary | Why it is untrusted |
| --- | --- |
| **Upstream task output / context** | Model output is data, not instructions. When piped downstream via `context_from`, it can carry injected directives, fabricated facts, or PII. |
| **External data pulled into a task** | Web scrapes, file uploads, user input, third-party API responses processed inside a task prompt. |
| **MCP tool documentation** | Tool descriptions and results from configured MCP servers (`mcp_servers` / `mcp_tools`) are external text that an engine may treat as authoritative. |
| **Machine-generated plans** | Plans produced by `maestro replan`, `maestro scaffold`, or the `watch --mode improve` loop are LLM output and should be treated as untrusted until audited. |
| **Cross-run memory / knowledge** | Lessons accumulated across runs (`.maestro-cache/`) can be poisoned by a prior compromised run. |

The clean-environment isolation in `runners.py` (`_ENV_ALLOWLIST`) sits on the
boundary between the operator's environment and the engine subprocess: only an
explicit allowlist of variables is inherited; everything else must be set via
`defaults.env` / `task.env`.

## Threats and Mitigations

### T1 -- Prompt injection via context

**Threat.** Output from an upstream task (or external data it ingested) contains
text designed to be interpreted as instructions by a downstream engine task,
hijacking its behavior. This is the central risk of any multi-step agent
pipeline.

**Mitigations Maestro ships:**

- **`context_trust: untrusted`** (per task) marks output as untrusted. Downstream
  consumers inherit taint transitively (`_compute_tainted_tasks` in
  `scheduler.py`); the `taint_detected` event records the source.
- **`control_flow_integrity: true`** (plan) and **`observation_block: true`**
  (per task) wrap upstream context in `<observation>` blocks
  (`_sandbox_observation` in `runners.py`) so the engine sees it as quoted data,
  not directives.
- **Injection-pattern stripping** (`_strip_injection_patterns`) removes common
  override phrases from untrusted context before it reaches the prompt.
- **Semantic firewall** -- when `firewall_model` is set on the plan, a second-pass
  LLM screen runs over untrusted/consolidated context.
- **`allowed_tools`** (per task) restricts which tools the engine may use, so a
  successful injection has a smaller blast radius. This is why the audit flags
  untrusted context without `allowed_tools` (`SEC023`).
- **`honeypot: true`** plants decoy values that only injected instructions would
  touch; access is surfaced via the `honeypot_triggered` event.
- **`guard_command` / `verify_command`** give a deterministic gate over output
  before it is trusted downstream.

Relevant audit rules: `SEC015` (when-expression reads raw output), `SEC016`
(raw `context_from` from an engine task without a guard), `SEC017` (external-data
upstream without `context_trust`), `SEC018` (inherited taint without a sanitizer),
`SEC019` (untrusted context without a honeypot), `SEC023` (untrusted context
without `allowed_tools`).

### T2 -- Memory / knowledge poisoning

**Threat.** A compromised or low-quality run writes misleading "lessons" into
cross-run memory, which then steers future runs toward the attacker's goal --
a slow, persistent attack that survives across invocations.

**Mitigations Maestro ships:**

- Memory records carry **provenance and trust labels** (`memory.py`); writes are
  surfaced via the `memory_write` event.
- **Instructionality scoring** rejects memory that reads as instructions rather
  than observations; **poisoning alerts** (`knowledge_poison_alert` event)
  flag statistically anomalous entries.
- **Consolidation safety gates** apply injection-pattern stripping, an optional
  firewall pass, and an instructionality threshold before any consolidated
  lesson is injected into a prompt; buckets dominated by untrusted evidence are
  rejected.
- Deep context chains without a token budget are flagged (`SEC010`) because they
  silently grow the surface where poisoned context can hide.

### T3 -- Secret leakage

**Threat.** API keys, tokens, or credentials end up in plan YAML, prompts, task
logs, run manifests, or HTML reports -- and then in a shared artifact store or a
pasted log.

**Mitigations Maestro ships:**

- **`secrets:` / `secrets: auto`** declares values to redact from logs,
  manifests, and reports; `--mask-secrets` enforces redaction at runtime
  (`_mask_secrets` in `runners.py`).
- **`output_redact`** (per task) scrubs PII/secret patterns from a task's output
  before it is passed downstream via `context_from`.
- **Environment isolation** (`_ENV_ALLOWLIST`) keeps unrelated host environment
  variables out of engine subprocesses.

Relevant audit rules: `SEC003` (secret-looking env var not declared), `SEC005`
(hardcoded key pattern in a prompt), `SEC007` (`secrets:` declared but
`--mask-secrets` is a runtime requirement), `SEC014` (cloud credentials with no
`secrets` configuration), `SEC020` (PII-producing upstream consumed without
`output_redact`).

> Run artifacts under `.maestro-runs/` are sensitive by default -- they contain
> prompts, command lines, paths, and model output even when secret *values* are
> masked. `.maestro-runs/` is gitignored; treat it accordingly.

### T4 -- Destructive commands

**Threat.** A task (authored or generated) runs an irreversible operation --
`rm -rf`, `DROP TABLE`, `git reset --hard`, a force-push -- against the operator's
real workspace or data.

**Mitigations Maestro ships:**

- **`requires_approval: true`** pauses for interactive confirmation before a task
  runs; `--auto-approve` is an explicit opt-out.
- **`phantom_workspace: true`** runs the task against a throwaway copy and only
  commits results back deliberately (`phantom_commit` event).
- **`worktree: true`** isolates engine filesystem changes to a dedicated git
  worktree, merged back only on success.
- **Execution profiles** (`plan` / `safe` / `yolo`): `safe` strips dangerous
  flags and adds sandbox/approval gates; `yolo` is an explicit, audited choice.

Relevant audit rules: `SEC008` (destructive command without approval), `SEC021`
(destructive patterns without `phantom_workspace` or approval), `SEC002` /
`SEC009` / `SEC012` (yolo/bypass flags without approval, worktree, or with
fallback propagation).

### T5 -- Supply chain via plugins and dependencies

**Threat.** A custom engine plugin (or a transitive dependency) executes
arbitrary code on the runner with the same privileges as Maestro itself.

**Mitigations and posture:**

- Custom engine plugins loaded from the `maestro_cli.engines` entry-point group
  are **arbitrary Python with full trust** -- discovery-time validation improves
  error messages but is **not** a sandbox. Installing a plugin is a code-execution
  trust decision, exactly like installing any other package.
- `maestro doctor` reports discovered plugins so the operator can see what is
  active.
- The built-in engine registry cannot be overridden by entry points.
- Core runtime depends only on PyYAML; optional features are isolated behind
  extras. CI stays offline-first and excludes real-engine tests by default.

See [SECURITY_BASELINE.md](SECURITY_BASELINE.md) for plugin and CI hardening
guidance.

## Defense-in-Depth Tooling

Maestro exposes static and runtime controls the operator can apply across the
threat surface above:

- **`maestro audit <plan.yaml>`** -- static scanner with rules `SEC001`-`SEC023`
  (defined in `src/maestro_cli/audit.py`), grouped into nine risk categories.
  `--coverage` shows per-category coverage; `--fix` applies safe remediations for
  `SEC001` / `SEC003` / `SEC014`. Custom `audit_packs` add project-specific
  assertions.
- **`maestro check <plan.yaml>`** -- validate + audit in one pass with a single
  exit code; the recommended first-run / CI gate.
- **`maestro verify <run-path>`** -- verifies the SHA-256 hash chain over
  `events.jsonl` (`eventsource.py`); a `tampered` result means a run's event log
  was altered after the fact.
- **Runtime policy engine** (`policy.py`) -- declarative `policies:` rules
  evaluated at task dispatch over a whitelisted field set using a safe AST
  evaluator (never `eval`/`exec`); actions are `block` / `warn` / `audit`, with a
  `policy_violation` event.
- **Generated-plan gate** -- plans produced by `maestro replan` are run through
  the audit gate (`_evaluate_generated_plan_security`) before execution, so
  replan's machine-authored variants are not implicitly trusted. (`scaffold`
  only emits YAML and does not execute it; `watch --mode improve` re-runs the
  target plan -- audit those yourself with `maestro audit` / `maestro check`
  before use.)

## Known Non-Goals

Maestro deliberately does **not** attempt to:

- Sandbox the engine CLIs or their model providers -- they run with the
  operator's credentials by design.
- Prevent an operator from authoring a deliberately dangerous plan, installing
  an untrusted plugin, or passing yolo/bypass flags. These are trust decisions,
  surfaced by the audit rules, not vulnerabilities.
- Provide multi-tenant isolation. Maestro is single-operator and local-first.

## Reporting

Found a security issue? Report it privately via GitHub Security Advisories --
see the root [SECURITY.md](../SECURITY.md). Please do not open a public issue
for a vulnerability.
