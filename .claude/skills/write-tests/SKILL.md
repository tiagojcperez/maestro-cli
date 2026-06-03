---
name: write-tests
description: Write pytest tests for a Maestro CLI module, following the project's testing conventions. Use when implementing new features, fixing bugs, or backfilling test coverage.
argument-hint: "[module-name]"
tags: testing, pytest, coverage
triggers: write tests, pytest, test coverage, failing tests, regression
recommended-when: Use after a feature or bug fix needs focused pytest coverage aligned with Maestro's existing testing conventions.
recommended-chain: write-tests
---

Write tests for: $ARGUMENTS

## Setup
Ensure pytest is available:
```powershell
py -m pip install pytest
```

## Test File Location
```
tests/
├── conftest.py          # Shared fixtures
├── test_<module>.py     # Tests for src/maestro_cli/<module>.py
```

## Conventions

### File Structure
```python
from __future__ import annotations

import pytest
from pathlib import Path

from maestro_cli.<module> import <functions_to_test>
from maestro_cli.models import PlanSpec, TaskSpec  # as needed
from maestro_cli.errors import PlanValidationError  # as needed


# --- Fixtures ---

@pytest.fixture
def sample_plan_yaml(tmp_path: Path) -> Path:
    """Create a minimal valid plan YAML for testing."""
    content = """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
"""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


# --- Tests ---

class TestFunctionName:
    def test_happy_path(self) -> None:
        ...

    def test_edge_case(self) -> None:
        ...

    def test_error_case(self) -> None:
        with pytest.raises(PlanValidationError, match="expected message"):
            ...
```

### Key Patterns
- Use `tmp_path` for all file operations (never write to repo)
- Use `monkeypatch` to mock subprocess calls
- Use `capsys` to capture print output
- Group related tests in classes (`class TestLoadPlan:`)
- Test both success and failure paths
- Use `@pytest.mark.parametrize` for multiple inputs
- Derive cases from concrete source behavior, fixtures, and failure modes — not
  from generic "QA expert" roleplay

### Mocking Subprocess
```python
def test_execute_task(monkeypatch, tmp_path):
    def mock_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)
    # ... test execute_task ...
```

### Running Tests
```powershell
py -m pytest tests/ -v
py -m pytest tests/test_loader.py -v
py -m pytest tests/ -k "test_cycle_detection"
```

## Checklist
- [ ] conftest.py exists with shared fixtures
- [ ] Happy path tested
- [ ] Error/edge cases tested
- [ ] Subprocess calls mocked (no real external engine CLI invocations)
- [ ] tmp_path used for file I/O
- [ ] All tests pass: `py -m pytest tests/ -v`
