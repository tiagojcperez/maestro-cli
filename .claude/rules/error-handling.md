# Rule: Error Handling

## Scope
All Python files in `src/maestro_cli/`.

## Exception Types

### PlanValidationError
- **When**: Schema violations, invalid YAML structure, missing fields, bad values
- **Where**: Only raised in `loader.py`
- **Effect**: Immediate exit with code 1 (fail-fast validation)
- **Message format**: Include context — task ID, field name, what's wrong

```python
raise PlanValidationError(f"Task '{task.id}' depends on unknown task '{dep}'")
```

### TaskExecutionError
- **When**: Runtime errors during task preparation (missing prompt file, unsupported engine)
- **Where**: Only raised in `runners.py`
- **Effect**: Captured as `TaskResult(status="failed")`
- **Message format**: Include task ID and specific error

```python
raise TaskExecutionError(f"Task '{task.id}' prompt_file not found: {task.prompt_file}")
```

## Error Handling Patterns

### In Loader (loader.py)
- Raise `PlanValidationError` immediately on any schema issue
- Validate EVERYTHING before returning `PlanSpec`
- Never return partially-valid plans
- Import validation (E024-E029): circular imports, missing files, duplicate prefixes, invalid prefix chars, approval misconfiguration
- Resilience validation (E030-E031): invalid escalation lists, invalid fallback engine/model configuration

### In Runners (runners.py)
- Wrap command building in try/except → return failed `TaskResult`
- Subprocess failures → capture exit_code in `TaskResult`
- Timeouts → `TaskResult(status="failed", exit_code=124)`
- Pre-command failures → `TaskResult(status="failed")`, skip main command
- Judge evaluation errors → `JudgeResult(verdict="error")`, graceful degradation (E107)
- Failure classification: `_classify_failure()` categorizes errors into `FailureCategory` for smart retry feedback
- Workspace index build errors → graceful fallback to empty context (E108)
- Workspace extraction LLM errors → graceful fallback to empty context (E109)
- Workspace brief LLM errors → graceful fallback to empty context (E110)

### In Scheduler (scheduler.py)
- Failed tasks with `allow_failure=true` → `soft_failed` (dependents proceed)
- Failed tasks with `allow_failure=false` → `failed` (dependents skipped)
- `fail_fast=true` → skip ALL remaining tasks on first failure
- Approval denied → `TaskResult(status="skipped", message="Approval denied")`
- Non-interactive approval → auto-skip (don't block)
- Never re-raise exceptions from `Future.result()` — capture in TaskResult

### In CLI (cli.py)
- Top-level try/except catches all exceptions → `print(f"[maestro] error: {exc}")` + exit 1
- Never let stack traces reach the user

## Anti-Patterns
- Do NOT use bare `except:` — always specify the exception type
- Do NOT use `assert` for validation (use explicit raise)
- Do NOT swallow exceptions silently (always log or record in TaskResult)
- Do NOT raise exceptions from inside the ThreadPoolExecutor callback
