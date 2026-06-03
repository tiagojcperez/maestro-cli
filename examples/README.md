# Maestro CLI Examples

This directory contains example plans and supporting files that demonstrate
Maestro CLI features. Most engine examples are illustrative (they reference
`claude`, `codex`, etc. and need an engine CLI on PATH plus credentials) -- so
validate or dry-run them first. The one fully self-contained example that runs
with no engine and no API key is `demo_plan.yaml`.

## Start here: `demo_plan.yaml`

`demo_plan.yaml` is the engine-free, no-API-key starting point. Every task is a
plain shell command (`python -c "print('...')"`), so it validates and runs end
to end on a clean machine:

```
maestro validate examples/demo_plan.yaml
maestro run      examples/demo_plan.yaml
```

It runs four tasks (a -> b, c -> d) and finishes with a `4 ok` summary. Run
artifacts land under `.maestro-runs/` (gitignored). Use it to confirm your
install works before reaching for the engine-backed examples.

## Example plans

| File | What it shows |
|------|---------------|
| `demo_plan.yaml` | Minimal 4-task shell DAG (a -> b/c -> d). Engine-free, runs to `4 ok` with no API key. |
| `advanced-features.yaml` | `prompt_md_file`, `pre_command`, `verify_command`, `max_retries`, `allow_failure`, templates, `env`, resume. |
| `context-mode-demo.yaml` | Inter-task context with `context_mode: map_reduce` and `summarized` feeding code-review / QA tasks. |
| `context-passing.yaml` | Passing upstream output downstream via `context_from` and `{{ task-id.stdout_tail }}` template vars. |
| `dynamic-group-demo.yaml` | `dynamic_group: true` -- a planner task generates sub-tasks at runtime, validated and executed as a nested DAG. |
| `tui_demo.yaml` | Long-running shell DAG with progress output, useful for exercising `--output tui`. |
| `windows-bash.yaml` | Windows-specific patterns: Git Bash list-format commands, MySQL paths, forward slashes, env isolation. |
| `pitfall-audit.yaml` | YAML anchors, `judge` (typed assertions + rubric + g_eval), `guard_command`, `context_mode: recursive`, budget controls, workspace indexing. |
| `scaffold-brief.yaml` | A brief consumed by `maestro scaffold` to generate a full plan with quality gates and build verification. |
| `library-brief.yaml` | A brief that pulls in a built-in workflow library (`library: rest-api`) and overrides/extends its tasks. |
| `custom-workflow-library.yaml` | An external workflow library file referenced from a brief via `library: <path>`. |

Suggested first commands for an engine-backed example (dry-run avoids spending
tokens):

```
maestro validate examples/context-passing.yaml
maestro run      examples/context-passing.yaml --dry-run
```

The scaffold/library briefs are inputs to `maestro scaffold`, not run directly:

```
maestro scaffold examples/scaffold-brief.yaml --validate --cost-check
maestro scaffold examples/library-brief.yaml -o generated-plan.yaml --validate
```

## `audit-packs/` -- workspace security baseline

Audit packs are reusable lists of workspace-level assertion rules
(`glob_exists`, `file_contains`, `file_regex`, package-present checks, etc.)
that run alongside the built-in SEC001-SEC023 checks in `maestro audit`.

| File | Purpose |
|------|---------|
| `audit-packs/security-baseline.yaml` | A reusable audit pack of baseline hygiene rules (BL001+: `.gitignore` present, secret-pattern checks, etc.). Reference it from a plan via `audit_packs:`. |
| `audit-packs/security-baseline-demo.yaml` | A plan that references the baseline pack and adds per-task `assert:` rules plus a `test-manifest` contract. |

```
maestro audit examples/audit-packs/security-baseline-demo.yaml
```

## `ci/` -- sample generated CI workflows

The `ci/` directory holds **sample outputs** of the CI generator. The supported
way to get a CI workflow is to generate it from your own plan rather than copy
these files:

```
maestro ci examples/demo_plan.yaml --provider github_actions -o .github/workflows/ci.yml
maestro ci examples/demo_plan.yaml --provider gitlab_ci      -o .gitlab-ci.yml
```

### Note on duplicated files

The current `ci/` snapshots overlap. Verified with `md5sum`:

- `github_actions.yml` and `github_actions_maestro.yml` are **byte-identical**
  (workflow name `Maestro CI`, Python `3.11`, validates `examples/demo_plan.yaml`).
- `github-actions.yml` is a **separate** variant (workflow name `Demo CI`,
  Python `3.12`, validates `plans/demo.yaml`) -- not a duplicate of the two above.
- `gitlab-ci.yml` and `gitlab_ci.yml` are **byte-identical**.

These duplicates are kept for now and may be deduplicated in a later pass. When
in doubt, regenerate with `maestro ci <plan> --provider ...` instead of editing
the samples by hand. Generated CI keeps validate/test lanes offline-first and
puts real-engine plan runs behind an explicit manual lane.
