# Security Baseline for v1.0.0

This document captures the minimum security posture expected for Maestro CLI
`1.0.x` usage and release engineering. It is an operational baseline, not a
formal security certification.

## Scope

The baseline covers four areas that are already present in the codebase and
release process:

- secrets handling in plans, logs, and run artifacts
- custom engine plugin trust boundaries
- CI hardening for offline and networked lanes
- opt-in handling for real-engine end-to-end tests

## Secrets Handling

- Prefer environment injection over inline plan secrets. Do not store API keys,
  tokens, or credentials directly in authored plan YAML.
- Use `secrets:` or `secrets: auto` so Maestro can redact matching values from
  logs, manifests, and reports. `secrets: auto` is a convenience, not a reason
  to stop naming sensitive values explicitly when the set is known.
- Treat `.maestro-runs/`, generated manifests, task logs, and HTML reports as
  sensitive-by-default artifacts. They may contain prompts, command lines,
  workspace paths, and model output even when secret values are masked.
- Do not print secrets intentionally inside task commands or prompts. Redaction
  reduces exposure but does not make deliberate secret echoing acceptable.
- In CI, scope secrets to the minimum job or environment that needs them and
  avoid exporting provider credentials into unrelated test lanes.

## Plugin Trust Boundaries

- Built-in engines are part of the reviewed Maestro codebase. Custom engine
  plugins loaded from the `maestro_cli.engines` entry-point group are arbitrary
  Python code with the same effective trust level as any other installed package.
- Only install plugins from repositories and package registries you already
  trust for code execution on the runner. Discovery-time validation improves
  error messages; it is not a sandbox.
- Do not allow untrusted pull requests to add or replace plugin packages in
  shared CI runners. Review dependency changes and lock plugin versions.
- Prefer running release and contract validation lanes with built-in engines
  only. Add custom plugins only in explicitly owned environments.

## CI Hardening

- Keep the default CI lane offline and deterministic: install from the checked
  out repository, run the normal unit/integration suite, and exclude
  `real_engine` tests by default.
- Pin third-party GitHub Actions or equivalent CI integrations to immutable
  versions or commit SHAs. Avoid broad write permissions for default jobs.
- Use least-privilege tokens. For GitHub Actions, keep `GITHUB_TOKEN`
  permissions narrow and do not grant secret-bearing jobs access unless they
  actually need networked engine execution.
- Separate networked or cost-incurring jobs from the normal test lane. Require
  explicit manual dispatch or equivalent approval before running them.
- Review artifact uploads. Do not publish full `.maestro-runs/` contents from
  secret-bearing jobs unless retention, audience, and masking behavior are
  understood and acceptable.

## Real-Engine Opt-In Tests

- `tests/test_e2e_real_engines.py` is intentionally opt-in and marked
  `real_engine`.
- Default local and CI runs should continue to use the normal suite without
  enabling `MAESTRO_RUN_REAL_ENGINE_TESTS=1`.
- Real-engine runs require explicit per-engine model environment variables, for
  example `MAESTRO_E2E_CODEX_MODEL` or `MAESTRO_E2E_OLLAMA_MODEL`, so paid or
  networked calls do not happen accidentally.
- Keep the real-engine lane manual or otherwise explicitly approved. It should
  use narrowly scoped secrets, budget awareness, and clear ownership.
- Ollama real-engine runs are local-engine tests, but they still require
  deliberate enablement and a runner that already has the requested model
  available.

## Release Expectation

For `1.0.x`, the release baseline is:

- secret masking stays enabled and documented
- plugin loading remains an explicit trust decision, not a sandbox promise
- default CI remains offline-first
- real-engine coverage remains opt-in rather than a mandatory default gate
