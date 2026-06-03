# Migrating from 0.x to v1.0.0

> Historical guide: this covers the now-historical `0.x` -> `1.x` migration. The current release line is `2.x` (see [../CHANGELOG.md](../CHANGELOG.md) and [ROADMAP.md](ROADMAP.md)); the steps below remain accurate for anyone still on `0.x`.

This guide is for users already running Maestro plans on `0.x`. v1.0.0 was a
stability release, not a schema reset. Most existing `version: 1` plans should
keep working without structural rewrites.

## What does not change

- Keep `version: 1` in authored plans
- Keep the same documented task/plan keys and the same built-in engine names:
  `codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`
- Keep using the existing stable run artifacts:
  `run_manifest.json`, `run_summary.md`, `events.jsonl`, `<task-id>.log`, and
  `<task-id>.result.json`

## Concrete migration checks

1. Stop relying on undocumented plan fields.

Fields and behaviors outside the v1 freeze are the main migration risk from
older `0.x` usage. In particular, move away from loader-accepted knobs that are
not part of the frozen authored `1.x` surface, such as:

- `secrets_auto`
- `requires_clean_worktree`
- `stdout_tail_lines`
- `append_system_prompt`

Use the documented authored form instead. The most common cleanup is:

```yaml
# Before
secrets_auto: true

# After
secrets: auto
```

2. Update any `replan` automation that still uses the old flag name.

`maestro replan` now uses `--model`.

```bash
# Before
maestro replan plan.yaml --analysis-model sonnet

# After
maestro replan plan.yaml --model sonnet
```

3. Move automation off scraped console output.

If your scripts parse human-readable stdout/stderr, switch them to stable files
or JSON-producing commands instead:

- Read `run_manifest.json` and task `*.result.json`
- Use `maestro doctor --json`
- Use `maestro diff --json`
- Use `maestro explain --json`
- Use `maestro status --json`
- Use `maestro eval --json`
- Use `maestro suggest --json`

Ignore unknown JSON keys so additive `1.x` releases do not break consumers.

4. Make cache path assumptions explicit.

There are two different cache locations:

- Task result cache: `<run-dir>/.cache`
- Workspace index cache: `<workspace_root>/.maestro-cache/index/`

If you built cleanup scripts around a single cache directory, split them now.

5. Keep real-engine tests opt-in.

Default test runs should remain:

```bash
python -m pytest tests/ -q
```

Only enable the real-engine harness intentionally:

```bash
MAESTRO_RUN_REAL_ENGINE_TESTS=1 \
MAESTRO_E2E_CODEX_MODEL=5-mini \
python -m pytest tests/test_e2e_real_engines.py -m real_engine -q
```

For local Ollama coverage, also set `MAESTRO_E2E_OLLAMA_MODEL` and make sure the
model is already present locally.

## New extension and release tooling

v1.0.0 documents two implemented extension/release surfaces that may already be
useful during migration:

- `maestro ci` generates GitHub Actions or GitLab CI starter YAML from a plan
- Custom engines can be loaded from the `maestro_cli.engines` entry-point group

If you want a concrete CI starting point during migration:

```bash
maestro ci plan.yaml --provider github --output .github/workflows/maestro.yml
maestro ci plan.yaml --provider gitlab --output .gitlab-ci.yml
```

Treat that output as generated starter config with first-class cross-platform
validation coverage, not as a frozen contract. The explicit `validate_linux`,
`validate_windows`, `validate_macos`, `test_linux`, `test_windows`, and
`test_macos` lanes are the intended migration baseline, while provider aliases
and exact YAML layout may still change in `1.1.0+`. Today that includes the
current manual real-engine jobs `maestro_real_engine` and
`run_maestro_real_engine`.

The currently documented `maestro ci` flags are:

- `--provider github_actions|gitlab_ci|github|gitlab`
- `--output`
- `--workflow-name`
- `--python-version`
- `--test-command`

The generated YAML now also starts with a small platform matrix comment block
so the Windows/macOS/Linux mapping is visible without reading the whole file.
Use the matrix as the quick coverage check:

- GitHub Actions: `linux`, `windows`, and `macos` are all required lanes.
- GitLab CI: `linux` is default-on; `windows` and `macos` stay opt-in until you
  provide matching tagged runners.
- Windows shell guidance: prefer YAML list-format commands for Bash or
  path-sensitive tools, not bash-only string commands such as
  `command: "bash -lc \"python -m pytest -q\""`.

Keep forward-slash paths inside Git Bash snippets such as `/c/project`, and
keep generated CI entrypoints shell-neutral with `python -m ...` and
`maestro ...`.

Both are available today, but neither surface is frozen by the v1.0.0 public
contract. If you adopt them in production, pin Maestro conservatively and expect
refinement in `1.1.0+`.

## Tooling posture in v1.0.0

- Strict mypy exists, but only for the allowlisted files in `pyproject.toml`
- The current strict mypy command is `python -m mypy`
- `maestro-benchmark` exists for deterministic local benchmarking
- Security guidance is documented in `docs/SECURITY.md` and
  `docs/SECURITY_BASELINE.md`
- No built-in security scanner or mandatory security audit gate is frozen into
  the v1.0.0 contract

Those tools are part of the release workflow around v1.0.0, but stricter gates
for plugin API stability, broader mypy coverage, and benchmark-based release
thresholds are intentionally deferred to `1.1.0+`.
