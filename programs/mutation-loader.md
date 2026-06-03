# Mutation Testing Program — loader.py

## Goal
Raise the mutation score for `src/maestro_cli/loader.py` to ≥ 90%.
Mutation score = percentage of mutants killed by the test suite.

## Module Overview
`loader.py` is the YAML parsing, validation, and plan construction core (~900 lines).
Key functions:
- `load_plan()` — top-level entry: reads YAML, resolves imports, expands matrix, validates
- `validate_plan()` — runs all E001-E056 validation rules; raises `PlanValidationError` on violations
- `_expand_matrix_tasks()` — Cartesian product expansion for `matrix:` tasks
- `_to_judge_spec()` — parses and validates `judge:` blocks (method, criteria, thresholds)
- `_resolve_imports()` — resolves `imports:` entries with prefix namespacing and cycle detection
- `_dag_detect_cycle()` — DFS cycle detection on dependency graph
- `compute_plan_density()` — (v1.14.0+) computes S_complex = exp(S_node + 2·S_edge + S_depth)
- Various `_to_*` coercion helpers: `_to_str_dict`, `_to_str_list`, `_to_float`, `_to_bool`, etc.

Test files:
- `tests/test_loader.py` — main loader unit tests (add new tests here)
- `tests/test_validation_warnings.py` — W-code warning tests (do not modify)

## Mutation Operator Guide
Common operators cosmic-ray uses on Python code:

| Operator | Changes | Example | Kill with |
|----------|---------|---------|-----------|
| `BinaryOperatorReplacement` | `+` ↔ `-`, `*` ↔ `/` | `depth + 1` → `depth - 1` | Assert depth increments correctly |
| `ComparisonOperatorReplacement` | `<` ↔ `<=`, `>` ↔ `>=`, `==` ↔ `!=` | `max_retries > 3` | Test exact boundary values (0, 3, 4) |
| `NumberReplacer` | literal → 0, 1, -1 | `100` → `0` | Assert context_budget_tokens min is enforced |
| `BooleanReplacer` | True ↔ False | `allow_failure=False` default | Test default is False, not True |
| `ReturnValueReplacement` | return X → return None/0 | `return plan_spec` → `return None` | Assert load_plan() returns PlanSpec |
| `StringReplacer` | string → `""` | error code `"E001"` → `""` | Assert error messages contain the code |

## Test Strategy

### 1. Validation boundary tests (high value)
- `max_retries` = 0, 1, 3 (valid) vs 4, -1 (E012)
- `max_parallel` = 1, 10 (valid) vs 0, -1 (E012)
- `context_budget_tokens` = 100 (valid) vs 99, 0 (E019)
- `judge.pass_threshold` = 0.0, 1.0 (valid) vs 1.01, -0.01 (E020)
- `budget_warning_pct` = 0.01, 1.0 (valid) vs 0.0, 1.01 (E023)
- `judge.timeout_sec` = 10, 60 (valid) vs 9, 0 (E020)
- `judge.quorum` = 2, 5 (valid) vs 1, 0 (E054)

### 2. Dependency graph tests
- Single-node cycle (self-dependency → E016)
- Two-node cycle (A→B→A → E004)
- Chain of 3+ dependencies (valid, no cycle)
- Diamond dependency (A→B, A→C, B→D, C→D → valid)

### 3. Matrix expansion tests
- Single key, 2 values → 2 tasks with IDs `parent@key=val`
- Two keys → Cartesian product (2×2 = 4 tasks)
- `{{ matrix.KEY }}` substituted in prompt and verify_command
- matrix + engine → valid; matrix + group → E062 (if tested)

### 4. Import resolution tests
- Valid import with prefix → task IDs get namespaced
- Circular import (A imports B, B imports A → E025)
- Missing import file → E025/E026
- Duplicate prefix → E027
- Invalid prefix chars → E028

### 5. Coercion helper tests
- `_to_bool`: "true"/"false"/1/0/None → correct booleans
- `_to_float`: "1.5"/1/None → correct floats or None
- `_to_str_list`: string → [string], list → list, None → []

### 6. Error message content tests
- PlanValidationError messages include the task ID
- PlanValidationError messages include the error code (E001, E004, etc.)
- validate_plan raises on unknown depends_on reference (E005)

## Constraints
- Only modify `tests/test_loader.py`
- Do NOT modify source files or `tests/test_validation_warnings.py`
- New tests must pass on unmodified source
- Use `tmp_path` for all file writes (never write to repo root)
- Use `pytest.raises(PlanValidationError)` for error path tests
- Prefer `@pytest.mark.parametrize` for boundary variations
