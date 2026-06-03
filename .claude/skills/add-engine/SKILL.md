---
name: add-engine
description: Add support for a new AI engine to Maestro CLI, or decide when a plugin is a better fit than a new built-in engine. Use when extending engine support beyond the current built-ins.
disable-model-invocation: true
argument-hint: "[engine-name]"
tags: engines, plugins, runtime
triggers: add engine, new engine, engine support, plugin, model provider
recommended-when: Use when extending built-in engine support, wiring a new provider, or deciding whether a plugin should stay outside the frozen core.
recommended-chain: add-engine -> write-tests
---

Add support for engine `$ARGUMENTS`.

## First Decision: Built-In Engine or Plugin?

Prefer a plugin when the new engine can live behind the `maestro_cli.engines`
entry-point group without changing the frozen core behavior.

Prefer a built-in engine only when at least one of these is true:
- it must be available out of the box
- it needs execution-profile integration in the core runtime
- it needs first-class doctor, cache, cost, or CLI support

## Built-In Engine Checklist

### 1. Models and schema
- Add the engine name to `EngineName` in `src/maestro_cli/models.py`
- Add a defaults slot in `PlanDefaults`
- Add any model aliases / pricing / context-window metadata if applicable

### 2. Loader and validation
- Parse defaults in `src/maestro_cli/loader.py`
- Allow the engine in `validate_plan()`
- Add validation warnings if the engine has special caveats

### 3. Runtime
- Add argument normalization in `src/maestro_cli/runners.py`
- Add command construction and execution-profile handling
- Update cache hashing in `src/maestro_cli/cache.py`
- Update doctor probes in `src/maestro_cli/doctor.py`
- Update historical cost detection if needed (`src/maestro_cli/cost_backfill.py`)

### 4. Tests
- Runner command-building tests
- Loader validation tests
- Doctor coverage
- Cache-hash stability tests if engine args/models normalize specially

### 5. Documentation
- `README.md`
- `CLAUDE.md`
- `.claude` agent/rule/skill docs if they mention engine inventories
- `CHANGELOG.md` and versioning docs if the new engine changes public posture

## Plugin Path Checklist

- Add the plugin under `[project.entry-points."maestro_cli.engines"]`
- Provide a concrete `EnginePlugin` object
- Add `DoctorProbe` metadata if the executable can be probed
- Document the plugin as implemented-but-unfrozen if it is not part of the
  frozen `1.x` contract

## Guardrails
- Do not hardcode engine-specific logic in the scheduler
- Keep the scheduler engine-agnostic; runtime details belong in runners/doctor/cache
- Avoid documenting the plugin authoring API as frozen unless the contract docs say so
