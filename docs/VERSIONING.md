# Versioning Rules

## Versioning Scheme

Maestro CLI follows [Semantic Versioning 2.0.0](https://semver.org/).

```
MAJOR.MINOR.PATCH
```

The package is currently at **2.5.3**. The `2.x` line builds additively on the
stable surface that was frozen at `1.0.0`; the `2.0.0` through `2.5.3` releases
have all shipped (see `CHANGELOG.md`).

Maestro's bump policy is **deliberately looser than strict SemVer**: the PATCH
digit is the *default* release stream and MINOR is reserved for milestones (see
the bump table below). Strict SemVer would make every additive feature a MINOR,
which over-inflates MINORs and leaves the PATCH digit unused.

## Package Version vs Plan-Schema Version

Maestro has **two independent version numbers**. Do not conflate them.

| What | Where | Current value | Governed by |
|------|-------|---------------|-------------|
| **Package version** | `pyproject.toml`, `src/maestro_cli/__init__.py` (`__version__`) | `2.5.3` | SemVer (this document) |
| **Plan-schema version** | the `version:` key inside an authored plan YAML | `1` | Plan schema contract |

These are decoupled on purpose:

- The **package version** describes the released `maestro` tool. It moves under
  SemVer as features are added, fixed, or (rarely) broken. It went from `1.x` to
  `2.0.0` for an additive milestone (Python SDK + capability-based tool access),
  not because the plan schema changed.
- The **plan-schema version** is the contract for files you author. It is still
  `version: 1`. The loader rejects anything else with `[E002] Only version: 1 is
  supported` (`loader.py`, `plan.version != 1`).

So a `2.4.0` package still runs `version: 1` plans, and it will keep doing so for
the entire `2.x` line. A renumber to plan `version: 2` would be a deliberate,
documented schema break - it is not implied by any package major bump.

## What Changes Require Which Package Bump

| Bump | When | Examples |
|------|------|----------|
| **MAJOR** | Breaking changes to the frozen YAML schema, the documented CLI surface, or the frozen run artifacts | renumbering the plan schema to `version: 2`, removing a CLI flag, renaming a manifest key |
| **MINOR** | A **significant milestone**: a sizable new capability, a headline batch of features, or a change to an existing *documented* default/behaviour | a new context-mode milestone batch, a notable subsystem, promoting a behaviour change worth headlining |
| **PATCH** | The **default stream** — bug fixes, docs, performance, tests, internal refactoring with no public-surface break, **and** small backward-compatible *opt-in* additive features that don't change an existing documented default | fix cost parsing, tighten validation, a new opt-in `context_mode` (e.g. `codebase_map`), an internal indexed retriever with an env opt-out (e.g. FTS5 knowledge ranking, `MAESTRO_KNOWLEDGE_FTS`) |

Litmus test for an additive change: small, opt-in, and behaviour-neutral for
existing users (or with a documented opt-out) → **PATCH**. A headline capability
or a shift to a documented default → **MINOR**. Most releases are PATCHes.

Breaking changes to the frozen public surfaces are reserved for the next MAJOR
release. Within a MAJOR line (e.g. all of `2.x`), the frozen surfaces stay
backward compatible and growth is additive.

## The v1 Frozen Contract (still authoritative)

The stable public contract was first frozen at `1.0.0` and is defined in
[docs/V1_API_FREEZE.md](V1_API_FREEZE.md). That document remains the normative
description of the stable surface, and it still applies under `2.x` because the
`2.x` releases extended that surface additively rather than breaking it.

The freeze covers three machine-relevant surfaces:

- authored `version: 1` plans using documented keys and value families
- the documented `maestro` command and flag surface
- the on-disk run artifacts intended for automation, including
  `run_manifest.json`, `run_summary.md`, `events.jsonl`, `<task-id>.log`, and
  `<task-id>.result.json`

For regression testing, treat the titled contract sections in
`docs/V1_API_FREEZE.md` as the exact source of truth:

- Stable YAML Schema
- Stable CLI Surface
- Stable Run Artifacts
- Compatibility Promise for `1.x`
- Explicit Non-Goals for `v1.0.0`
- Migration Posture from `0.x` to `1.0.0`

The freeze is intentionally narrower than the full feature set. A feature can
ship without being frozen; it only becomes part of the stability promise once it
is listed in the freeze document. Treat old roadmap bullets and README summaries
as informative context, not as normative contract text.

## Promotions and Deferrals (current status)

Some surfaces were originally implemented outside the `1.0.0` freeze. Their
current status:

- **`maestro ci` provider surface** - promoted into the frozen contract in
  `v1.31.0`. The stable command surface and structural YAML guarantees are
  defined in [docs/V1_API_FREEZE.md](V1_API_FREEZE.md).
- **Custom engine plugins** via the `maestro_cli.engines` entry-point group -
  promoted into the frozen contract in `v1.31.0`. The frozen authoring surface
  is documented in [docs/V1_API_FREEZE.md](V1_API_FREEZE.md).
- **Strict mypy coverage** - now runs over the **full `src/maestro_cli/`
  package** under `strict = true` (`pyproject.toml` `[tool.mypy]`,
  `files = ["src/maestro_cli/"]`). This is no longer an allowlist of a few files.
  Type-checking scope and tooling posture, however, are still a development
  concern rather than a frozen public contract.
- **Benchmark thresholds and security-process gating** - still deferred. The
  `maestro-benchmark` entry point is available for deterministic local
  measurement, and security guidance lives in `docs/SECURITY.md` and
  `docs/SECURITY_BASELINE.md`, but no benchmark threshold or built-in security
  scanner is a mandatory release gate.

Each item above appears exactly once: either it is promoted into the freeze, or
it remains deferred. There is no item that is both.

## What Is Frozen (practical posture)

The practical release posture is:

- keep authored plans on `version: 1`
- preserve documented command names and option spellings
- preserve frozen artifact filenames and documented JSON keys
- allow additive growth in minor releases

Concrete details live in [docs/V1_API_FREEZE.md](V1_API_FREEZE.md). This file
states policy; the freeze document states the exact regression targets.

## Opt-In Real-Engine Policy

Default test and CI lanes remain offline-first.

- `tests/test_e2e_real_engines.py` is marked `real_engine`
- it skips unless `MAESTRO_RUN_REAL_ENGINE_TESTS=1`
- individual lanes also require explicit model env vars such as
  `MAESTRO_E2E_CODEX_MODEL` or `MAESTRO_E2E_OLLAMA_MODEL`
- generated CI keeps real-engine execution on an explicit manual lane rather
  than the default validate/test path

Mandatory default real-engine gating is intentionally deferred.

## Migration Posture From 0.x

The `1.0.0` freeze was a contract freeze, not a schema reset, and `2.x` did not
reset it either.

- existing documented `version: 1` plans should keep working under `2.x`
- undocumented `0.x` knobs should be removed from authored plans
- automation should read stable files and JSON outputs instead of scraping
  human-readable console text
- scripts should treat unknown JSON keys as additive and ignore them
- where old roadmap bullets were broader than the final freeze, the freeze
  document wins for compatibility and regression testing

Use [docs/MIGRATING_TO_V1.md](MIGRATING_TO_V1.md) for the concrete cleanup
steps, including:

- replacing old `maestro replan --analysis-model` usage with `--model`
- separating task result cache from workspace index cache
- keeping real-engine testing explicitly opt-in

## Historical Pre-1.0 Rules

Through `0.15.0`, Maestro used pre-1.0 SemVer expectations:

| Bump | When |
|------|------|
| **0.MINOR.0** | New features, schema changes, and breaking internal/public changes |
| **0.x.PATCH** | Bug fixes, tests, docs, and non-breaking internal work |

Breaking YAML, CLI, and output changes were allowed in `0.MINOR` releases and
had to be documented in `CHANGELOG.md`.

## Version Locations

The release version has **two source-of-truth locations** that must always be
updated together:

1. `pyproject.toml`
2. `src/maestro_cli/__init__.py`

The following files are usually **documentation mirrors** of the current
release and should be audited on every bump if they mention the current
version explicitly:

1. `README.md`
2. `CHANGELOG.md`
3. `CLAUDE.md`
4. `CODEX.md`
5. `docs/FEATURE-READINESS.md`
6. `docs/ROADMAP.md`
7. Local project memory/snapshot files such as `MEMORY.md`

## Release Checklist

1. Update the version in both source-of-truth locations.
2. Update `CHANGELOG.md` with the released version number and date.
3. Audit the version mirrors (`README.md`, `CLAUDE.md`, `CODEX.md`, `docs/FEATURE-READINESS.md`, `docs/ROADMAP.md`, local `MEMORY.md` if used).
4. Run the offline test suite (`python -m pytest tests/ -q`).
5. Run `python -m mypy` (strict over the full `src/maestro_cli/` package).
6. Run any explicitly chosen opt-in release checks.
7. Commit and tag the release.

## Per-Release History (living sources)

This file no longer maintains a per-version "Done In" log; those lists drift out
of date. The authoritative, living sources for what shipped in each release are:

- **[CHANGELOG.md](../CHANGELOG.md)** - the full, dated release history
  (every `MAJOR.MINOR.PATCH` from the `0.x` line through `2.4.0`).
- **[docs/ROADMAP.md](ROADMAP.md)** - planned and in-flight work, organized by
  phase.

Consult those documents for feature-by-feature release detail.
