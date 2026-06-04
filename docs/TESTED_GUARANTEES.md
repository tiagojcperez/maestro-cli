# Tested Guarantees

This document states **what is actually tested, at which level, and what runs in
default CI** — so you can calibrate trust before depending on Maestro. It is
deliberately conservative: if something is not continuously proven, it says so.

It complements the [v1 stability contract](V1_API_FREEZE.md) (what the API
*promises*) and the [security baseline](SECURITY_BASELINE.md) (the trust model).

## Test levels

| Level | What it means | Cost / where it runs |
|-------|---------------|----------------------|
| **Unit (mocked)** | The function's logic is exercised with `subprocess.run`, the filesystem, time, and network mocked. No real CLI is ever invoked. | Free. Every push, full matrix. |
| **Integration** | Multiple modules wired together — loader → scheduler → runner — with engine calls still mocked. Verifies command construction, DAG ordering, context passing, artifact shapes. | Free. Every push, full matrix. |
| **Real-engine (E2E)** | A real provider CLI is invoked end-to-end and its output is parsed for real. Needs the CLI installed and (for cloud engines) credentials + spend. | Opt-in (`MAESTRO_RUN_REAL_ENGINE_TESTS=1`). Only the free local Ollama leg runs in CI. |

Default CI is **offline-first and deterministic**: engine calls are mocked, so a
green build proves Maestro's orchestration logic, not provider availability.

## Default CI surface (every push)

Runs on Python **3.11 / 3.12 / 3.13** plus a **Windows** lane:

- Full test suite (**12k+ tests**, engine calls mocked)
- Strict `mypy` over the whole `src/maestro_cli/` package
- Documentation lint (release-hygiene checks)
- CodeQL (`security-extended`)
- Coverage upload to Codecov

Plus a separate **weekly + on-demand** lane that runs the **free local Ollama**
real-engine round-trip (`real-engine.yml`) — the one E2E leg cheap enough to run
unattended.

## Per-engine coverage

All seven engines are covered at the unit and integration levels (command
building, model-alias resolution, profile application, cost/token parsing). They
differ in whether a **real end-to-end** test exists and whether it runs in CI.

| Engine | Unit (mocked) | Integration | Real-engine E2E test | In default CI |
|--------|:-------------:|:-----------:|:--------------------:|:-------------:|
| `ollama` | ✅ | ✅ | ✅ round-trip | ✅ weekly + on-demand (free) |
| `codex` | ✅ | ✅ | ✅ round-trip | ⚪ opt-in (paid) |
| `claude` | ✅ | ✅ | ⚪ manual only | ❌ |
| `gemini` | ✅ | ✅ | ⚪ manual only | ❌ |
| `copilot` | ✅ | ✅ | ⚪ manual only | ❌ |
| `qwen` | ✅ | ✅ | ⚪ manual only | ❌ |
| `llama` | ✅ | ✅ | ⚪ manual only | ❌ |

- **✅ round-trip** — a dedicated test in `tests/test_e2e_real_engines.py` drives
  a real plan through the CLI and asserts on the parsed manifest/result.
- **⚪ manual only** — no dedicated E2E test ships yet; you can still run these
  engines, and their command construction and output parsing are unit/integration
  tested, but end-to-end correctness against the live CLI is verified manually,
  not continuously. Adding round-trips here (behind the same opt-in flag) is the
  main outstanding test-coverage gap.

To run the opt-in real-engine tests locally:

```bash
MAESTRO_RUN_REAL_ENGINE_TESTS=1 MAESTRO_E2E_OLLAMA_MODEL=llama3.2:1b \
  python -m pytest tests/test_e2e_real_engines.py -m real_engine -k ollama
```

## Per-area coverage

| Area | Unit | Integration | Real-engine |
|------|:----:|:-----------:|:-----------:|
| DAG scheduling / dependency resolution | ✅ | ✅ | n/a |
| Shell (`command`) tasks | ✅ | ✅ | n/a |
| Context passing (9 modes) | ✅ | ✅ | partial |
| Budgets / cost / token tracking | ✅ | ✅ | partial |
| Quality gates (judge / verify / guard / assert) | ✅ | ✅ | partial |
| Retries / fallback / circuit breakers | ✅ | ✅ | n/a |
| Secret masking | ✅ | ✅ | n/a |
| Policy engine (safe AST evaluation) | ✅ (fuzzed) | ✅ | n/a |
| Security audit (SEC rules) | ✅ | ✅ | n/a |
| Cache / event sourcing / blame | ✅ | ✅ | n/a |
| Python SDK surface (`__all__`, `py.typed`) | ✅ | ✅ | n/a |

The policy engine is additionally **fuzzed** (`tests/test_policy_fuzz.py`) to
confirm the safe AST evaluator rejects `eval`/`exec`/`open`/`__import__`, dunder
access, comprehensions, lambdas, and arithmetic/bitwise abuse — it never calls
`eval()`/`exec()`.

## What Maestro guarantees

- DAG ordering and dependency semantics (including `fail_fast`, `allow_failure`,
  `when` conditions).
- Retry / fallback / escalation / circuit-breaker behaviour.
- The documented `version: 1` plan schema and run-artifact shapes
  (`run_manifest.json`, `run_summary.md`, `events.jsonl`, `*.log`,
  `*.result.json`) — frozen by the [v1 contract](V1_API_FREEZE.md).
- Best-effort secret masking in logs and manifests.
- Deterministic, replayable, hash-chained run logs (`maestro verify`).

## What Maestro does not guarantee

- Deterministic LLM output — providers are non-deterministic by nature.
- Continuous compatibility with every provider CLI — only Ollama is smoke-tested
  in CI; the rest is verified at unit/integration level and manually E2E.
- Pricing accuracy — pricing tables are best-effort and overridable via env vars.
- The safety of arbitrary user-authored shell commands — you own the plans you
  run. Use `maestro audit` to catch dangerous patterns before running.

## Outstanding gaps (tracked honestly)

1. **Real-engine round-trips for claude / gemini / copilot / qwen / llama** —
   currently manual; should become opt-in tests behind the existing flag.
2. **Real-engine CI breadth** — only the free Ollama leg runs unattended; paid
   engines stay opt-in by deliberate cost choice (see `SECURITY_BASELINE.md`).
