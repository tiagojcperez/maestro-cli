# Maestro CLI — Codex Instructions

Project instructions for OpenAI Codex CLI tasks.

## Project

- **Package**: `maestro-cli` (import: `maestro_cli`)
- **Path**: `src/maestro_cli/`
- **Version**: 2.4.0
- **Python**: >=3.11
- **Deps**: PyYAML only (stdlib for everything else)
- **Tests**: `py -m pytest tests/ -q` (~11.3K tests; offline by default)

## Architecture

DAG scheduler for AI execution plans. YAML in → validated PlanSpec → parallel task execution → JSON/log artifacts out.

Key modules:
- `cli.py` — 27 argparse subcommands (incl. `--set KEY=VALUE` template injection on run/replan/watch)
- `models.py` — 60+ dataclasses (PlanSpec, TaskSpec, TaskResult, ScoreRecord, etc.)
- `loader.py` — YAML parsing, validation (E001-E072 + runtime E100-E110), cycle detection, matrix expansion
- `runners.py` — subprocess execution, verify/retry, judge evaluation, secrets masking
- `scheduler.py` — DAG scheduling (ThreadPoolExecutor), context budget, BM25 intent filtering + RRF fusion
- `policy.py` — safe AST-based policy engine (block/warn/audit)
- `blame.py` — causal failure attribution via dependency graph walk
- `routing.py` — semantic model routing (auto model selection)
- `audit.py` — plan security scanner (SEC001-SEC023)
- `eventsource.py` — hash-chained event log with tamper detection
- `errors.py` — PlanValidationError, TaskExecutionError with error codes

## Code Style

- `from __future__ import annotations` in ALL files (FIRST import)
- PEP 604 unions: `str | None` (NEVER `Optional[str]`)
- Built-in generics: `list[str]`, `dict[str, str]` (NEVER `List`, `Dict`)
- `Literal[...]` for finite value sets
- Dataclasses for all data models (NEVER plain dicts)
- `field(default_factory=...)` for mutable defaults
- Full type annotations on ALL function signatures (params + return)
- f-strings for all interpolation
- `pathlib.Path` for all path operations (NEVER `os.path`)
- `encoding="utf-8"` on every `open()`, `read_text()`, `write_text()`
- Private helpers: `_snake_case`; constants: `_UPPER_SNAKE_CASE`
- Console output: `print(f"[maestro] ...")`
- No classes except `@dataclass` and `Exception` subclasses

## Error Handling

- `PlanValidationError` — raised only in `loader.py`, immediate exit
- `TaskExecutionError` — raised only in `runners.py`, captured in TaskResult
- Validation errors = fail early. Runtime errors = captured in TaskResult
- Timeouts → exit_code=124
- Never use bare `except:`, never use `assert` for validation
- Never raise exceptions inside ThreadPoolExecutor callbacks

## Testing Conventions

- pytest (not unittest), `tmp_path` for all file ops
- ALWAYS mock `subprocess.run` — never invoke real engine CLIs
- `pytest.raises(ExceptionType, match="pattern")` for error tests
- `@pytest.mark.parametrize` for variations
- Test both success AND failure paths
- One test file per source module: `test_<module>.py`

## YAML Plan Schema

- `version: 1` (required)
- `name` (required, non-empty)
- `tasks` (required, non-empty list)
- Each task: exactly one of `command`, `engine`, or `group`
- Engines: `codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`, `llama`
- Engine tasks need prompt source: `prompt`, `prompt_file`, or `prompt_md_file` + `prompt_md_heading`
- `depends_on` must reference existing task IDs, no cycles
- `context_from` must reference `depends_on` entries (except wildcard `"*"`)

## Key Types (models.py)

```
EngineName = Literal["codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"]
ExecutionProfile = Literal["plan", "safe", "yolo"]
TaskStatus = Literal["success", "failed", "soft_failed", "skipped", "dry_run"]
ContextMode = Literal["raw", "summarized", "map_reduce", "recursive", "layered", "selective", "structural", "council", "knowledge_graph"]
RetryStrategy = Literal["constant", "linear", "exponential"]
PolicyAction = Literal["block", "warn", "audit"]
RoutingStrategy = Literal["cost_optimized", "quality_first", "balanced"]
```

## When Adding a New Field

1. Add to dataclass in `models.py` with safe default
2. Parse in `loader.py`
3. Validate in `validate_plan()` if constraints exist
4. Wire into runtime path
5. Add tests before documenting

## When Adding a New Subcommand

1. Add `_cmd_<name>()` handler in `cli.py`
2. Add subparser in `_build_parser()`
3. Wire in `main()` dispatch
4. Add tests in `tests/test_cli.py`
5. Update `docs/CLI_REFERENCE.md`

## Important Notes

- `.maestro-runs/` is gitignored — never commit run outputs
- Template variable `{{ workspace_root }}` resolves to empty string if not set
- Windows paths in YAML: use forward slashes
- `pre_command` failures prevent main command from running
- `verify_command` runs after main command; failure marks task as failed
- `max_retries` (0-3) retries main command + verify on failure (not pre_command)
- `allow_failure: true` → `soft_failed` status (doesn't block dependents)
- `--yolo` normalized to engine-specific bypass flags automatically
- Run `py -m pytest tests/ -q` before committing any changes

## Detailed Documentation

- Full plan schema and authoring guide: `docs/PLAN_GUIDE.md`
- CLI reference with all flags: `docs/CLI_REFERENCE.md`
- Model alias tables: `docs/MODELS.md`
- Agent operations manual: `docs/AGENT_OPS.md`
- Agent role catalog: `AGENTS.md`
