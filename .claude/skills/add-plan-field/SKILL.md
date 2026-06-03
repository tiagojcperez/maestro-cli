---
name: add-plan-field
description: Add a new configuration field to the Maestro CLI `version: 1` plan schema. Use when extending plan, defaults, or task options and wiring them through validation, execution, and docs.
disable-model-invocation: true
argument-hint: "[field-name] [scope: plan|defaults|task]"
tags: schema, yaml, plans
triggers: plan field, yaml field, schema field, task field, defaults field
recommended-when: Use when the `version: 1` plan schema needs a new field plus loader, validation, runtime, and docs wiring.
recommended-chain: add-plan-field -> write-tests
---

Add a new field `$ARGUMENTS` to the Maestro CLI plan schema.

## 1. Confirm the Scope
- **Plan level**: `PlanSpec` fields such as `max_parallel`, `max_cost_usd`, `imports`
- **Defaults level**: `PlanDefaults` fields such as `timeout_sec`, `retry_delay_sec`, `context_budget_tokens`
- **Task level**: `TaskSpec` fields such as `guard_command`, `context_compact`, `requires_approval`

Use `src/maestro_cli/models.py` as the source of truth for the current schema.

## 2. Update the Dataclasses
Add the field with the narrowest correct type and a sensible default:

```python
@dataclass
class TaskSpec:
    # ...
    new_field: str | None = None
```

Rules:
- use `X | None`, not `Optional[X]`
- use `field(default_factory=...)` for mutable defaults
- keep defaults compatible with existing `version: 1` plans

## 3. Parse It in `loader.py`
- read the YAML value in the right section
- use existing coercion helpers when possible (`_to_str_list`, `_to_str_dict`, `_to_int_or_none`, etc.)
- preserve `None` vs missing semantics if they matter downstream
- if the field interacts with the exact-one-of execution modes, keep the
  `command` / `engine` / `group` invariant intact

## 4. Validate It
Add loader validation when the field has constraints:
- type/shape limits
- allowed values
- dependencies on other fields
- mutual exclusion / required-with rules

Examples:
- context settings that require `context_from`
- prompt settings that require `engine`
- fields that need `workspace_root`

## 5. Wire It Into Runtime Behavior
Touch only the modules that actually need the field:
- `scheduler.py` for plan-level orchestration or dependency behavior
- `runners.py` for execution, command construction, retry, or prompt behavior
- `cli.py` only if the field must surface in command UX
- `doctor.py`, `cache.py`, `cost_backfill.py`, or `plugins.py` if the field
  affects those systems

## 6. Test It
Add focused tests for:
- valid plan with the field
- missing field uses the expected default
- invalid usage fails with a clear validation error
- runtime behavior changes, if the field is behavior-affecting

Typical targets:
- `tests/test_loader_validation.py`
- `tests/test_runners.py`
- `tests/test_scheduler.py`
- narrower feature-specific files if they already exist

## 7. Update Docs
If the field is public-facing, update the smallest relevant set:
- `README.md`
- `CLAUDE.md`
- `.claude/rules/yaml-schema.md`
- any skill/agent doc that teaches the affected workflow

If the field is implemented but intentionally not frozen, document that posture
instead of presenting it as part of the stable `1.x` contract.

## Checklist
- [ ] Dataclass field added with type annotation and default
- [ ] Loader parsing handles missing, explicit null, and normal values
- [ ] Validation covers invalid combinations
- [ ] Runtime logic uses the field in the correct module
- [ ] Tests cover valid, missing, and invalid cases
- [ ] Public docs reflect the real contract posture
