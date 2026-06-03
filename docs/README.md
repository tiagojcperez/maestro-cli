# Maestro CLI Documentation Index

This is the documentation map for Maestro CLI, a Python CLI orchestrator that
runs multi-step AI execution plans as a dependency-resolved DAG across 7
engines. The docs below are grouped by audience so a newcomer can find the
right starting point.

New here? Read the top-level project README first, then run the no-API-key
demo plan described under Getting Started.

---

## Getting started

| Doc | What it covers |
|-----|----------------|
| [../README.md](../README.md) | Project overview, install, and the storefront router for everything else |
| [../examples/demo_plan.yaml](../examples/demo_plan.yaml) | Minimal shell-only plan you can run with no API key as a first execution |
| [../examples/README.md](../examples/README.md) | Guide to the example plans (context modes, dynamic groups, audit packs, Windows bash) |

---

## Plan authoring

| Doc | What it covers |
|-----|----------------|
| [PLAN_GUIDE.md](PLAN_GUIDE.md) | The plan schema: tasks, context modes, reliability, contracts, and judge gates |
| [PLAYBOOK.md](PLAYBOOK.md) | Curated recipes by task type, anti-patterns, and a cost checklist |
| [MODELS.md](MODELS.md) | Engine model alias tables, model routing, pricing, and CWE judge profiles |
| [PITFALLS.md](PITFALLS.md) | Windows-specific and general gotchas to avoid when authoring plans |
| [AGENT_OPS.md](AGENT_OPS.md) | Operations manual for AI agents driving Maestro: decision trees and checklists |

---

## Reference

| Doc | What it covers |
|-----|----------------|
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Full flag tables for the 27 maestro subcommands |
| [V1_API_FREEZE.md](V1_API_FREEZE.md) | The frozen v1.0.0 public contract and what must stay backward-compatible |
| [VERSIONING.md](VERSIONING.md) | Semantic versioning rules and the current 1.x contract scope |
| [MIGRATING_TO_V1.md](MIGRATING_TO_V1.md) | Migration guide for users coming from 0.x plans |
| [ROADMAP.md](ROADMAP.md) | Planned features and the phased v2 roadmap |
| [FEATURE-READINESS.md](FEATURE-READINESS.md) | Historical feature readiness snapshot (planning reference, not a live ledger) |

---

## Contributing and policy

| Doc | What it covers |
|-----|----------------|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Dev setup, testing, mypy, benchmarks, and CI expectations |
| [../CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) | Community standards for participation in the project |
| [../SECURITY.md](../SECURITY.md) | How to report a vulnerability and the project's security policy |
| [SECURITY.md](SECURITY.md) | Security posture and trust-boundary notes for running plans and plugins |
| [SECURITY_BASELINE.md](SECURITY_BASELINE.md) | Minimum security baseline expected for 1.0.x usage and release engineering |
| [../CHANGELOG.md](../CHANGELOG.md) | Full release history |

---

## Project AI instructions

These files instruct the AI coding agents that work on the Maestro CLI codebase
itself; they are not user-facing usage docs.

| Doc | What it covers |
|-----|----------------|
| [../CLAUDE.md](../CLAUDE.md) | Full codebase instructions for Claude Code |
| [../CODEX.md](../CODEX.md) | Compact project instructions for Codex CLI tasks |
| [../AGENTS.md](../AGENTS.md) | Agent role catalog with the collaboration map |

---

## A note on internal notes

Private working notes and drafts are kept outside the published repository and
are intentionally not linked here. They are gitignored and are not part of the
public docs set.
