# Contributing to Maestro CLI

Thank you for your interest in contributing to Maestro CLI!

This is the single canonical contributing guide. (The copy under `docs/` is now
a stub that points back here.)

## Getting Started

### Prerequisites

- Python 3.11, 3.12, or 3.13
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/tiagojcperez/maestro-cli.git
cd maestro-cli

# Install in editable mode (CLI only -- PyYAML is the single runtime dependency)
pip install -e .

# Or install with optional extras for development
pip install -e ".[live,web,tui,agui,mcp,otel]"

# pytest is NOT a declared dependency -- install it explicitly for the test suite
pip install pytest
```

### Optional extras

The core install depends only on PyYAML. Each extra pulls in the libraries for
one optional surface:

| Extra  | Adds |
|--------|------|
| `live` | Rich live progress table (`maestro run --output live`) |
| `web`  | FastAPI + uvicorn Web UI (`maestro ui`) |
| `tui`  | Textual TUI (`maestro run --output tui`) |
| `agui` | AG-UI protocol adapter |
| `mcp`  | MCP protocol server (`maestro mcp-server`) |
| `otel` | OpenTelemetry OTLP exporter |

Install one at a time (`pip install -e ".[live]"`) or all together
(`pip install -e ".[live,web,tui,agui,mcp,otel]"`).

## Development Workflow

### Running Tests

```bash
# Run all tests (offline, ~11.3K tests)
python -m pytest tests/ -q

# Run a specific test file
python -m pytest tests/test_loader.py -v

# Run tests matching a pattern
python -m pytest tests/ -k "test_cycle" -v
```

All tests must pass before submitting changes. The default test suite runs
entirely offline -- no real engine CLIs or API keys are needed.

### Real-engine tests (opt-in)

The real-engine harness lives in `tests/test_e2e_real_engines.py` and is marked
`real_engine`. It skips unless `MAESTRO_RUN_REAL_ENGINE_TESTS=1`, and each engine
lane has its own explicit model env var so contributors do not make networked or
paid calls by accident. CI stays offline-first; these tests are skipped by
default.

Local opt-in:

```bash
export MAESTRO_RUN_REAL_ENGINE_TESTS=1
export MAESTRO_E2E_CODEX_MODEL=5-mini
# Optional local-only lane if Ollama is installed and the model is already pulled:
export MAESTRO_E2E_OLLAMA_MODEL=llama3
python -m pytest tests/test_e2e_real_engines.py -m real_engine -q
```

CI opt-in:

```bash
MAESTRO_RUN_REAL_ENGINE_TESTS=1 \
MAESTRO_E2E_CODEX_MODEL=5-mini \
python -m pytest tests/test_e2e_real_engines.py -m real_engine -q
```

Use the explicit manual CI lane (`workflow_dispatch` / manual job) to inject the
same env vars only for the real-engine run. If `MAESTRO_E2E_OLLAMA_MODEL` is set
in CI, the runner must also have `ollama` installed, the daemon reachable, and
that model already pulled; otherwise the local-engine test skips cleanly.

### Type Checking (mypy)

The repository ships a strict mypy configuration in `pyproject.toml`. Strict
checking runs over the full `src/maestro_cli/` package
(`[tool.mypy]` `files = ["src/maestro_cli/"]`, `strict = true`) -- not a reduced
file subset.

```bash
python -m mypy
```

### Documentation Lint

```bash
python scripts/doc_lint.py
```

### Code Style

- `from __future__ import annotations` in every Python file
- PEP 604 unions: `str | None` (not `Optional[str]`)
- `pathlib.Path` for all file operations (not `os.path`)
- f-strings for string formatting (not `.format()` or `%`)
- `encoding="utf-8"` on all file I/O
- No classes except `@dataclass` and `Exception` subclasses
- Console output: `print(f"[maestro] ...")`

See [.claude/rules/code-style.md](.claude/rules/code-style.md) for the full
style guide.

### Type Safety

- Full type annotations on all function signatures (params + return)
- `Literal` types for enum-like values
- `field(default_factory=...)` for mutable defaults in dataclasses
- Avoid `Any` except at serialization boundaries

See [.claude/rules/type-safety.md](.claude/rules/type-safety.md) for details.

### Testing Conventions

- pytest (not unittest)
- Use `tmp_path` for all file operations -- never write to the repo
- Use `monkeypatch` to mock subprocess and environment
- Mock `subprocess.run` for engine execution tests -- never call real CLIs
- Use `@pytest.mark.parametrize` for multiple input variations
- Test both success and failure paths

See [.claude/rules/testing.md](.claude/rules/testing.md) for the full guide.

## Project Structure

```
src/maestro_cli/
  cli.py            # argparse CLI (27 subcommands)
  models.py         # Dataclasses
  loader.py         # YAML parsing + validation
  runners.py        # Command building + task execution
  scheduler.py      # DAG scheduler
  plugins.py        # Custom engine plugin discovery
  tui/              # Textual TUI app (optional: pip install -e ".[tui]")
  live.py           # Rich Live output (optional: pip install -e ".[live]")
  web/              # FastAPI dashboard (optional: pip install -e ".[web]")
  ...               # See CLAUDE.md for the full module list
```

**Data flow**: YAML -> `loader.load_plan()` -> `PlanSpec` ->
`scheduler.run_plan()` -> `runners.execute_task()` -> `.maestro-runs/<run>/`

Maestro shells out to 7 supported engines (`codex`, `claude`, `gemini`,
`copilot`, `qwen`, `ollama`, `llama`) plus raw shell commands.

## Adding Features

### New Task Field

1. Add the field to `TaskSpec` (or `PlanDefaults`) in `models.py` with a default
2. Parse it in `loader.py` using the appropriate coercion helper
3. Validate it in `validate_plan()` if it has constraints
4. Wire it into the runtime path (`runners.py` or `scheduler.py`)
5. Add tests in the narrowest relevant test file
6. Update `CLAUDE.md` and `README.md`

### New Engine

1. Create an `EnginePlugin` entry point (see `plugins.py`)
2. Add model aliases and pricing tables to `runners.py`
3. Add a `DoctorProbe` for `maestro doctor` checks
4. Add tests with mocked subprocess calls
5. Update documentation

### New CLI Subcommand

1. Add the parser in `_build_parser()` in `cli.py`
2. Implement the `_cmd_<name>()` dispatch function
3. Update `README.md`, `CLAUDE.md`, and the CLI engineer agent

## Error Handling

- **Validation errors** (`PlanValidationError`): raised in `loader.py`, fail-fast
- **Runtime errors** (`TaskExecutionError`): raised in `runners.py`, captured in `TaskResult`
- Error codes: E001-E072 (validation, with gaps), E100-E110 (runtime)
- Never let stack traces reach the user -- `cli.py` catches all exceptions

See [.claude/rules/error-handling.md](.claude/rules/error-handling.md).

## CI Generator

Use `maestro ci <plan.yaml>` to generate starter CI config. See
[docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) for all flags.

```bash
# Print GitHub Actions YAML to stdout
maestro ci plan.yaml

# Write GitHub Actions YAML
maestro ci plan.yaml --provider github --output .github/workflows/maestro.yml

# Write GitLab CI YAML with custom metadata
maestro ci plan.yaml --provider gitlab --workflow-name "Maestro CI" --python-version 3.12
```

Generator behaviour:
- Validates the plan first
- Generates `linux`, `windows`, and `macos` validate/test lanes
- Adds a manual opt-in real-engine lane (not mandatory)
- Uses shell-neutral commands in generated steps

`maestro ci` is supported but its rendered YAML shape is intentionally outside
the frozen `1.x` contract.

## Cross-Platform CI

- `examples/ci/github-actions.yml` covers `linux`, `windows`, and `macos` explicitly
- `examples/ci/gitlab-ci.yml` keeps `linux` on by default with `windows` and `macos` as opt-in
- Generated CI keeps steps shell-neutral with `python -m ...` and `maestro ...`
- For Windows lanes, keep task commands in YAML list format when Bash or path-sensitive tools are involved
- Inside Git Bash snippets on Windows, use forward-slash paths (`/c/project`); on macOS and Linux use regular paths

Preferred Windows Bash pattern in plans:

```yaml
command:
  - "C:\\Program Files\\Git\\bin\\bash.exe"
  - -lc
  - "python -m pytest -q"
```

## Benchmarks

Use the built-in benchmark entry point for deterministic local measurements:

```bash
maestro-benchmark --iterations 5 --warmups 1 --task-count 200 --max-parallel 8
maestro-benchmark --case loader --case cache --case scheduler
maestro-benchmark --case replan_pruning --case replan_population --case replan_novelty --case replan_guidance
```

`maestro-benchmark` exercises `loader`, cache hashing, dry-run scheduling, and
deterministic integration scenarios:

- `replan_pruning` compares exhaustive multi-variant search against the same search with preloaded failure history and reports saved candidate simulations.
- `replan_population` compares single-shot replan with N>1 tournament search and reports the selected-fitness gain.
- `replan_novelty` compares selection with novelty scoring disabled vs enabled and reports whether the chosen candidate changed.
- `replan_guidance` compares replan search without the Phase 2 bridge against the same search with `failure_pattern` / `model_pattern` guidance plus similar-history bootstrap, and reports the saved simulations plus the resulting guidance bonuses.

It is diagnostic tooling, not a required release threshold.

## Security

Security guidance lives in:

- [docs/SECURITY.md](docs/SECURITY.md) -- threat model and mitigation guidance
- [docs/SECURITY_BASELINE.md](docs/SECURITY_BASELINE.md) -- v1.0.0 security posture

The plan security scanner ships rules SEC001-SEC023 (defined in
`src/maestro_cli/audit.py`). Key principles: secret masking is documented,
plugin loading is a trust decision (not sandboxed), default CI is offline-first,
and real-engine testing is opt-in. No mandatory security audit gate is frozen
into the v1.0.0 contract.

## Key Principles

- **Minimal dependencies** -- PyYAML only; optional extras for live/web/tui/agui/mcp/otel
- **Local-first persistence** -- run artifacts go to JSON + logs; cross-run knowledge and memory use a local per-plan SQLite store (`.maestro-cache/memory/`). No server, no external database
- **Engine-agnostic** -- the scheduler is generic; engine specifics stay in runners
- **Fail-fast validation** -- catch errors in the loader, not at runtime
- **Strong typing** -- dataclasses everywhere, ~95% type hint coverage

## Commit Messages

- Focus on the "why", not the "what"
- Keep the first line concise (under 72 characters)
- Use imperative mood: "Add feature" not "Added feature"

## Questions?

- Check [docs/PITFALLS.md](docs/PITFALLS.md) for common gotchas
- Check [CLAUDE.md](CLAUDE.md) for comprehensive project documentation
- Open an issue for bugs or feature proposals
</content>
</invoke>
