# Rule: Testing

## Scope
All test files in `tests/`.

## Framework
- pytest (not unittest)
- No external test dependencies beyond pytest itself

## Conventions

### File Organization
- One test file per source module: `test_<module>.py`
- Shared fixtures in `conftest.py`
- Group related tests in classes: `class TestLoadPlan:`
- Test methods named `test_<what_is_being_tested>`

### Fixtures
- Use `tmp_path` for all file operations (NEVER write to the repository)
- Use `monkeypatch` to mock subprocess and environment
- Use `capsys` to capture print() output
- Fixtures that create YAML files should return `Path` objects
- Reusable plan YAML stored as module-level string constants

### Mocking
- ALWAYS mock `subprocess.run` for engine execution tests
- Never invoke real engine CLIs in tests (`codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`, `llama-cli`)
- Mock at the lowest level possible (`subprocess.run`, not `execute_task`)

### Assertions
- Use plain `assert` statements (not `self.assertEqual`)
- Use `pytest.raises(ExceptionType, match="pattern")` for error testing
- Use `@pytest.mark.parametrize` for multiple input variations
- Test both success AND failure paths for every function

### What to Test
- Loader: valid parsing, missing fields, invalid values, cycles, duplicate IDs
- Runners: command building per engine, profile application, prompt loading, timeout
- Scheduler: dependency resolution, parallel execution, fail-fast, soft failures
- Utils: path resolution, template rendering, markdown extraction
- CLI: argument parsing, exit codes, error output

### What NOT to Test
- Third-party library internals (PyYAML parsing, argparse behavior)
- Python language features
- Private helper functions directly (test through public functions)

### Running
```powershell
py -m pytest tests/ -v                     # All tests
py -m pytest tests/test_loader.py -v       # Single module
py -m pytest tests/ -k "test_cycle"        # By name pattern
py -m pytest tests/ --tb=short             # Shorter tracebacks
```

## Type Hints in Tests
- Test functions should have `-> None` return type
- Fixture return types should be annotated
- `from __future__ import annotations` in test files too
