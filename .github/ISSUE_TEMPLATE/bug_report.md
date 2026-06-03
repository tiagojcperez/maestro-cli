---
name: Bug report
about: Report a reproducible problem in Maestro CLI
title: "[bug] "
labels: bug
assignees: ''
---

## What happened

A clear, concise description of the bug.

## Reproduction

The minimal plan YAML and/or command that triggers it. Use a fenced block.

```yaml
# plan.yaml (trimmed to the smallest case that reproduces)
```

```sh
maestro run plan.yaml --dry-run
```

## Expected vs actual

- **Expected:** what you thought would happen.
- **Actual:** what actually happened (include relevant log lines or the
  failing task's `.result.json` / `run_summary.md` snippet).

## Environment

```text
# maestro --version
<paste output>

# maestro doctor
<paste output>
```

- OS:
- Python version (3.11 / 3.12 / 3.13):

## Additional context

Anything else that helps (engine in use, relevant config, screenshots).
