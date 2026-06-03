# Rule: Agent Routing

## Scope
All `.claude/agents/*` role files, `.claude/skills/*` workflows, and Maestro
plans that use `agent:`.

## Why This Exists
- Agent roles help on alignment-heavy work such as review, QA, safety, and
  user-facing polish.
- The March 19, 2026 PRISM paper found that richer expert personas can hurt
  accuracy on knowledge-retrieval and discriminative tasks, especially when the
  role prompt is long and always-on.
- In this repo, agent files should behave as **selective operational lenses**,
  not blanket "act like an expert" personas.

Paper: https://arxiv.org/html/2603.18507v1

## Core Policy

### 1. `agent:` is opt-in, not default
- Add `agent:` only when specialized judgment or alignment is the main risk.
- Do **not** add `agent:` just to make a prompt sound more senior or expert.

### 2. Prefer constraint-first prompts
Put information in this order:
1. exact files and commands
2. source of truth (`models.py`, `loader.py`, parser, tests, manifests, logs)
3. acceptance criteria / invariants
4. role activation, only if it still helps

### 3. Keep role activation minimal
- Use the smallest amount of role text that changes behavior in a useful way.
- Avoid generic identity framing such as "senior", "world-class", or
  "10x engineer".
- Longer persona text is higher-risk on precision tasks.

## Routing Heuristics

### Precision-first: prefer no agent or minimal activation
Use plain, source-grounded prompts for:
- schema or field inventory
- exact command/flag lookup
- log and manifest reading
- API/contract recall
- dependency tracing
- diff summarization
- cost arithmetic
- extracting current behavior from tests or code

### Alignment-first: agent activation usually helps
Use agent roles for:
- code review and severity triage
- QA coverage design and acceptance gating
- security review and trust-boundary analysis
- plan synthesis and DAG shaping
- user-facing CLI/help/docs polish
- TUI UX review and interaction design

### Mixed tasks: split the work
When a task needs both factual recall and specialist judgment, split it into:
1. precision-first analysis without `agent:`
2. specialist implementation/review with `agent:`
3. deterministic verification

This is the closest practical analog to PRISM's binary routing in Maestro plans.

## Verification Rules
- Any agent-driven recommendation that changes code, schema, or plan behavior
  must be checked against the current source of truth before acceptance.
- Prefer deterministic checks (`verify_command`, `guard_command`, `assert:`)
  before subjective LLM judging.
- If comparing a baseline output to a specialist output, keep the specialist
  result only when it clearly improves the task and still passes deterministic
  checks.
- For subjective A/B review, use conservative pairwise comparison with swapped
  ordering when practical to reduce verbosity/position bias.

## Model Notes
- Reasoning-distilled or already specialized models may benefit less from extra
  role steering.
- When in doubt, bias toward raw constraints + verification instead of more
  persona text.
