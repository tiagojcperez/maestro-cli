---
name: create-plan
description: Create a new YAML execution plan for the Maestro CLI orchestrator from scratch. Use when the user needs a new plan to orchestrate a multi-step AI workflow.
disable-model-invocation: true
argument-hint: "[goal or description]"
tags: plans, scaffold, yaml
triggers: new plan, plan yaml, scaffold, workflow plan, execution plan
recommended-when: Use when starting a brand-new Maestro workflow and you need the initial DAG, task types, and refinement checklist.
recommended-chain: create-plan -> write-tests
---

Create a new Maestro CLI execution plan for: $ARGUMENTS

## Recommended Workflow

1. Write a brief YAML.
2. Run `maestro scaffold brief.yaml -o plan.yaml --validate --cost-check`.
3. Treat the scaffold output as a starting point, then refine it by hand.
4. Run `maestro validate plan.yaml` and `maestro run plan.yaml --dry-run`.

## Important Current Caveat

`maestro scaffold` still routes its built-in task types through Claude models:

| `task_type` | Engine | Model | Agent |
|---|---|---|---|
| `shell`, `branch-setup`, `build-verify` | shell | -- | -- |
| `trivial-fix` | `claude` | `haiku` | -- |
| `implementation`, `complex-implementation` | `claude` | `sonnet` | -- |
| `code-review` | `claude` | `sonnet` | `code-reviewer` |
| `qa-verification` | `claude` | `sonnet` | `qa-engineer` |
| `security-audit` | `claude` | `opus` | `security-engineer` |

If the environment is Codex-first or has no Claude tokens, use scaffold only for
topology and naming, then rewrite:

- `engine` / `model`
- `context_mode`
- `judge`
- any review tasks that would otherwise depend on Claude-specific flows

For Codex-only plans, prefer `context_mode: raw`, deterministic
`verify_command`/`guard_command`, and explicit `tags` so reruns can stay cheap.

## Agent Routing Pattern

- Do not treat `agent:` as mandatory. It is a selective routing tool.
- Leave `agent:` out of precision-first tasks such as schema inventory, command
  lookup, manifest/log reading, diff summary, or exact contract extraction.
- Use `agent:` when the task mainly needs specialist judgment: implementation
  under repo conventions, code review, QA, security review, plan shaping.
- For mixed work, split the DAG:
  1. analyze facts without `agent:`
  2. implement/review with the relevant agent
  3. verify deterministically
- Avoid generic persona text like "you are a senior expert". Put files,
  invariants, and acceptance criteria in the prompt instead.
- If you compare a baseline output with an agent-shaped output, keep the
  specialist version only if deterministic checks still pass.

## Brief Template

```yaml
name: <descriptive-name>
goal: "<one-line goal>"
workspace_root: "C:/path/to/project"
branch_name: feature/<branch-name>
max_parallel: 3

tasks:
  - id: <task-id>
    description: "<what this task does>"
    task_type: implementation
    prompt_hint: "<short prompt seed>"

  - id: <next-task>
    description: "<what this task does>"
    task_type: implementation
    depends_on: [<task-id>]
    prompt_hint: "<short prompt seed>"
```

## Manual Refinement Checklist

- Keep `version: 1`.
- Each task must define exactly one of `command`, `engine`, or `group`.
- Use forward slashes in paths.
- Use `.claude/rules/agent-routing.md` when deciding whether a task needs `agent:`.
- Prefer `prompt_md_file` + `prompt_md_heading` for long prompts.
- Add `verify_command` for any task with a deterministic outcome.
- Add `guard_command` when validating agent stdout is cheaper than a judge.
- Use `context_from` explicitly; add `context_compact: true` when upstream logs are noisy.
- Use `tags` so slices can run with `--tags` / `--skip-tags`.
- Set `cache: false` only for tasks that really must always re-run.
- Use `requires_approval` only for truly risky steps.
- Use `secrets: auto` or explicit secret names at plan level.
- Add `max_cost_usd` and `budget_warning_pct` when cost control matters.

## Manual Plan Skeleton

```yaml
version: 1
name: <descriptive-name>
workspace_root: C:/path/to/project
max_parallel: 2
fail_fast: false
max_cost_usd: 20.0
budget_warning_pct: 0.7
secrets: auto

defaults:
  env:
    PYTHONUTF8: "1"
  timeout_sec: 900
  retry_delay_sec: [5.0, 15.0]
  context_budget_tokens: 5000
  stdout_tail_lines: 80
  edit_policy: efficient
  codex:
    model: "5.4"
    reasoning_effort: medium
    args: ["--dangerously-bypass-approvals-and-sandbox"]

tasks:
  - id: implement-feature
    description: "Implement the feature"
    engine: codex
    prompt_md_file: docs/prompts.md
    prompt_md_heading: "Implement Feature"
    verify_command:
      - "py"
      - "-m"
      - "pytest"
      - "tests/test_feature.py"
      - "-q"
    tags: [impl]

  - id: review-output
    description: "Review the implementation outcome"
    depends_on: [implement-feature]
    context_from: [implement-feature]
    context_mode: raw
    context_compact: true
    engine: codex
    prompt: |
      Review the upstream result:
      {{ implement-feature.stdout_tail }}
    tags: [review]

  - id: deploy-gate
    description: "Manual gate before deploy"
    depends_on: [review-output]
    requires_approval: true
    approval_message: "Proceed with deploy?"
    command:
      - "py"
      - "-c"
      - |
        print("approval reached")
    tags: [release]
```

## Schema Notes That Matter In Practice

- `context_mode: summarized`, `map_reduce`, and `recursive` add extra model work.
- `recursive` requires `workspace_root` and surfaces `{{ workspace_brief }}`.
- `matrix:` expands one authored task into multiple concrete tasks.
- `imports:` composes reusable task sets under prefixed IDs.
- `group:` runs a nested plan.
- `maestro ci` and custom engine plugins are implemented, but they are not part
  of the frozen `1.x` contract.

## Validation Sequence

```powershell
maestro validate plan.yaml
maestro run plan.yaml --dry-run
maestro explain plan.yaml
maestro status plan.yaml
```

## Engine Args Safety (CRITICAL — prevents wasted budget)

1. **Codex write access**: Codex defaults to read-only sandbox. Tasks that edit
   or create files need one of:
   - CLI-level: `maestro run plan.yaml --execution-profile yolo` (preferred)
   - YAML defaults: `defaults.codex.args: ["--dangerously-bypass-approvals-and-sandbox"]`
   - Per-task: `args: ["--dangerously-bypass-approvals-and-sandbox"]`
   - `--full-auto` gives `workspace-write` sandbox (edits OK, new files may fail)

2. **Never mix `fallback_engine` with engine-specific `args:`** — fallback
   inherits args from the primary engine. `--full-auto` (codex) crashes Claude.
   Either use `--execution-profile yolo` at CLI level or don't use `fallback_engine`
   on tasks with engine-specific args.

3. **First run of a new plan: consider `max_retries: 0`** — validates that flags,
   prompts, and verify commands work before burning budget on retries. Enable
   retries after first successful validation.

4. **Always test engine CLI flags locally** before putting them in YAML:
   ```
   codex exec --full-auto "print hello"
   claude --print --model sonnet "print hello"
   ```

## Final Checklist

- [ ] DAG is intentional and parallelism is real
- [ ] Every task has exactly one execution mode (`command`, `engine`, or `group`)
- [ ] Prompt sources and markdown headings are valid
- [ ] Verification is deterministic where possible
- [ ] Cost-sensitive plans avoid accidental Claude-dependent context/judge flows
- [ ] Codex tasks have write access (P16/P17: test flags locally first!)
- [ ] No `fallback_engine` on tasks with engine-specific `args:` (P15)
- [ ] `maestro validate` passes
- [ ] `maestro run --dry-run` matches expectations
