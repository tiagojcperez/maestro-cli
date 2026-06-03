---
name: create-dynamic-plan
description: Create a Maestro plan with dynamic_group tasks that generate sub-plans at runtime. Use when the user wants an LLM to analyze a problem and autonomously decompose it into sub-tasks.
disable-model-invocation: true
argument-hint: "[goal or description of what to accomplish]"
tags: planning, dynamic-group, decomposition
triggers: dynamic group, dynamic_group, sub-plan, subplan, autonomous decomposition
recommended-when: Use when the plan should synthesize a constrained nested DAG at runtime instead of spelling every task out up front.
recommended-chain: create-dynamic-plan -> write-tests
---

Create a Maestro CLI plan using `dynamic_group: true` for: $ARGUMENTS

## What is dynamic_group?

A task with `dynamic_group: true` runs in two phases:
1. **Phase 1**: Engine task generates a sub-plan as JSON (via `output_schema`)
2. **Phase 2**: Maestro validates and executes the sub-plan as a nested DAG

The generated sub-plan is **untrusted** — the runtime applies a strict 7-field
allowlist and forces `execution_profile: "safe"`.

## Prompting Policy

- Do not ask the planner to invent generic expert personas inside generated
  prompts.
- Generated task prompts should be constraint-first: exact files, required
  output, invariants, and acceptance criteria.
- If the dynamic workflow mixes factual discovery and specialist judgment,
  create an initial fact-finding task and then follow-up implementation/review
  tasks that consume those facts.
- Prefer explicit verification/review tasks over verbose "act like an expert"
  framing.

## Template

```yaml
version: 1
name: <plan-name>
workspace_root: "."
max_cost_usd: <budget>  # REQUIRED with dynamic_group

tasks:
  - id: <planner-task-id>
    engine: claude
    model: sonnet
    agent: dynamic-planner
    dynamic_group: true
    output_schema:
      type: object
      required: [name, tasks]
      properties:
        name: {type: string}
        tasks:
          type: array
          items:
            type: object
            required: [id, engine, prompt]
            properties:
              id: {type: string}
              engine: {type: string}
              prompt: {type: string}
              model: {type: string}
              description: {type: string}
              depends_on:
                type: array
                items: {type: string}
              tags:
                type: array
                items: {type: string}
    prompt: |
      Analyze the workspace and create a plan to: <GOAL>

      Context:
      - workspace_root: {{ workspace_root }}
      - Read relevant files before planning
      - Output valid JSON with a "tasks" array

      Each task needs: id, engine, prompt.
      Optional: model (haiku/sonnet), depends_on, description, tags.
      Use haiku for reads/simple edits, sonnet for implementation.

  - id: review
    engine: claude
    model: <review-model>
    depends_on: [<planner-task-id>]
    prompt: |
      Review all changes made by the dynamic sub-plan.
      Sub-plan results: {{ <planner-task-id>.stdout_tail }}
```

## Checklist

1. **`max_cost_usd` is set** — dynamic_group generates untrusted tasks with
   real cost. Without a budget cap, a confused LLM can generate expensive plans.
2. **`agent: dynamic-planner`** — the dedicated agent role instructs the LLM to
   only generate the 7 allowed fields and follow cost-aware planning rules.
3. **`output_schema` is defined** — required for `dynamic_group: true`. The
   schema validates the JSON structure before the sub-plan is built.
4. **Review task depends on the planner** — the dynamic sub-plan runs inside the
   planner task. The review task sees merged results via `{{ id.stdout_tail }}`
   (sub-task outputs) and `{{ id.output.sub_tasks }}` (structured summary).
5. **Prompt includes context** — tell the planner what to analyze and what the
   goal is. Use `{{ goal }}`, `{{ workspace_root }}`, and upstream context vars.
6. **No persona inflation** — longer roleplay text can hurt precision. Ask for
   concrete constraints, not identity performance.

## Security Notes

The runtime enforces these constraints on ALL dynamic sub-plans:
- **Allowlist**: only `id`, `engine`, `prompt`, `model`, `depends_on`,
  `description`, `tags` are read from LLM output. All other fields ignored.
- **Safe mode**: sub-plan always runs with `execution_profile: "safe"`
- **CFI**: `control_flow_integrity` forced `True` on sub-plan
- **Budget**: sub-plan receives **remaining** budget, not total
- **No recursion**: generated tasks cannot be `dynamic_group`
- **No commands**: only engine tasks allowed (shell commands stripped)
- **Task cap**: max 20 tasks (excess truncated)

## Validation

```powershell
maestro validate <plan.yaml>
maestro run <plan.yaml> --dry-run
maestro audit <plan.yaml>          # checks for SEC001 (no budget)
```
