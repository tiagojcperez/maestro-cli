# Mutation Testing Program — routing.py

## Goal
Raise the mutation score for `src/maestro_cli/routing.py` to ≥ 90%.
Mutation score = percentage of mutants killed by the test suite.

## Module Overview
`routing.py` provides semantic model routing for `model: auto` tasks.
Key functions:
- `resolve_auto_model()` — entry point; picks engine + model based on complexity
- `_score_task_complexity()` — scores a task 0.0–1.0 using tags, prompt length, deps, context mode, judge, DAG metadata
- `_tier_from_score()` — maps score to low/medium/high tier
- Routing tier tables per engine (claude, codex, gemini, copilot, qwen, ollama)

## Mutation Operator Guide
Common operators cosmic-ray uses on Python code:

| Operator | Changes | Example | Kill with |
|----------|---------|---------|-----------|
| `BinaryOperatorReplacement` | `+` ↔ `-`, `*` ↔ `/` | score + 0.4 → score - 0.4 | Assert score is higher, not lower |
| `ComparisonOperatorReplacement` | `<` ↔ `<=`, `>` ↔ `>=`, `==` ↔ `!=` | threshold < 0.5 | Test exact boundary values |
| `NumberReplacer` | literal → 0, 1, -1 | 0.4 → 0 | Assert non-zero tag bonus applied |
| `BooleanReplacer` | True ↔ False | return True → False | Test return value polarity |
| `ReturnValueReplacement` | return X → return None/0 | | Test function return is used correctly |

## Test Strategy
1. **Boundary tests** — score values that cross tier thresholds (low/medium/high)
2. **Tag signal tests** — security/architecture tags → high tier; trivial/typo → low tier
3. **Arithmetic tests** — verify score deltas are added, not subtracted
4. **Composition tests** — multiple signals combine correctly (not overriding each other)
5. **Engine-specific tests** — each engine's tier table is exercised

## Constraints
- Only modify `tests/test_routing.py`
- Do NOT modify source files
- New tests must pass on unmodified source
- Prefer `@pytest.mark.parametrize` for boundary variations
- No external mocking needed (pure functions)
