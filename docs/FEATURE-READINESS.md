# Feature Readiness Snapshot (Historical)

**Snapshot captured**: 2026-03-22 | **Repo reviewed**: 2026-04-09 | **Current repo state**: Maestro v2.4.0+main, ~11.3K tests in the latest full-suite run, ~48.6K Python LOC

This file is a planning snapshot, not a live feature ledger. The matrix and module ratios below preserve the historical readiness view used during the v1.24.0-v1.31.0 hardening window. For current shipped status, prefer [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](../CHANGELOG.md).

## How to read this

Each pending feature is scored on 4 dimensions:

| Dimension | Score | What it means |
|---|---|---|
| **Coverage** | A-D | LOC/test ratio of worst module involved (A ≤ 5, B ≤ 10, C ≤ 16, D > 16) |
| **Integration** | ●●● / ●●○ / ●○○ | Integration test coverage for the module interaction path |
| **History** | ✓ clean / ⚠ warn / ✗ fail | Historical failures in Sprint runs touching these modules |
| **Security** | — / 🔒 / 🔒🔒 | No security surface / touches security / core security feature |

**Composite Risk**: Low / Medium / High / Critical — derived from worst dimension.

---

## Risk Summary

> **Note:** the ASCII box below is the 2026-03-22 capture. Several items have since
> shipped — see the **Full Matrix** tables below for current status (Medium: 10 shipped /
> 0 remaining; High: 2 shipped / 2 remaining; Critical: 2 shipped / 1 remaining).

```
┌──────────────────────────────────────────────────────────┐
│  LOW RISK (11 shipped, 0 remaining)        ALL DONE ✓    │
│  ██████████████████████████████████████████████           │
│                                                          │
│  MEDIUM RISK (8 shipped, 2 remaining)              × 2   │
│  ██████████████████████████████████████                   │
│                                                          │
│  HIGH RISK (1 shipped, 3 remaining)                 × 3  │
│  ████████████████                                        │
│                                                          │
│  CRITICAL RISK (defer to 2.0 / experimental)        × 3  │
│  ████████████                                            │
└──────────────────────────────────────────────────────────┘
```

---

## Full Matrix

### Low Risk — ship with confidence

| Feature | Modules | Coverage | Integration | History | Security | Effort | Target |
|---|---|---|---|---|---|---|---|
| ~~Agent-Triggered Compression~~ | runners, scheduler | B/C | ●●○ | ✓ | — | S | **v1.24 DONE** |
| ~~Event-Driven Reminders~~ | runners | B | ●○○ | ✓ | — | S | **v1.24 DONE** |
| ~~Watch Step Counter~~ | watch | B | ●●○ | ✓ | — | S | **v1.24 DONE** |
| ~~Run Knowledge Expansion~~ | knowledge | A | ●●○ | ✓ | — | M | **v1.24 DONE** |
| ~~OTLP Exporter~~ | otel | — | ●○○ | ✓ | — | M | **v1.24 DONE** |
| ~~Honeypot Decoys (Kavach)~~ | runners, audit | B/A | ●●○ | ✓ | 🔒🔒 | S | **v1.24 DONE** |
| ~~Workflow Libraries~~ | scaffold | A | ●●○ | ✓ | — | S | **v1.25 DONE** |
| ~~Multi-Dimensional Eval~~ | eval | A | ●●○ | ✓ | — | M | **v1.25 DONE** |
| ~~Security Contracts (Envelope)~~ | eventsource | A | ●●○ | ✓ | 🔒 | M | **v1.25 DONE** |
| ~~CWE Security Profiles~~ | models | A | ●●○ | ✓ | 🔒 | S | **v1.25 DONE** |
| ~~Dual Verification (Worktrees)~~ | worktree | A | ●●○ | ✓ | 🔒 | S | **v1.25 DONE** |

### Medium Risk — ship with extra testing first

| Feature | Modules | Coverage | Integration | History | Security | Effort | Target |
|---|---|---|---|---|---|---|---|
| ~~Staged Progressive Compaction~~ | runners | B | ●●○ | ⚠ timeout | — | M | **v1.26 DONE** |
| ~~Trajectory-Level Guardrails~~ | scheduler | B | ●●○ | ✓ | 🔒 | M | **v1.26 DONE** |
| ~~Privacy-Aware Context Pipeline~~ | runners | B | ●○○ | ✓ | 🔒🔒 | M | **v1.26 DONE** |
| ~~Population-Based Search~~ | runners, scheduler | B | ●○○ | ✓ | — | M | **v1.28 DONE** |
| ~~Semantic Firewalls~~ | runners | B | ●○○ | ✓ | 🔒🔒 | M | **v2.3 DONE** |
| ~~Llama-CPP Engine~~ | runners, plugins | B | ●○○ | ✓ | — | M | **v2.1 DONE** |
| ~~Python SDK~~ | all | varies | ●○○ | ✓ | — | L | **v2.0 DONE** |
| ~~Skill Registry~~ | scaffold, cli | A | ●○○ | ✓ | — | M | **v1.27 DONE** |
| ~~Phantom Output Interception (Kavach)~~ | runners, audit | B/A | ●○○ | ✓ | 🔒 | M | **v1.26 DONE** |
| ~~Adaptive Temporal Routing (RuVector)~~ | routing, knowledge | A | ●●○ | ✓ | — | M-L | **v1.28 DONE** |

### High Risk — prototype in branch first

| Feature | Modules | Coverage | Integration | History | Security | Effort | Target |
|---|---|---|---|---|---|---|---|
| ~~MCP-Native Tool Orchestration~~ | runners, mcp_server | B | ●○○ | ✓ | 🔒🔒 | L | **v1.29 DONE** |
| A2A Engine + Server | new modules | — | ○○○ | ✓ | 🔒🔒 | L | v2.0 |
| ~~CI Agentic Workflows~~ | ci, watch, multi | A/B | ●○○ | ✓ | — | M | **v1.27 DONE** |
| Environment Drift Benchmark | benchmark, scheduler | A/B | ○○○ | ✓ | — | M | v2.0 |

### Critical Risk — defer to 2.0 / experimental gate

| Feature | Modules | Coverage | Integration | History | Security | Effort | Target |
|---|---|---|---|---|---|---|---|
| ~~Council Mode (Multi-Model)~~ | runners, scheduler | B | ○○○ | ✓ | 🔒 | L | **v2.1 DONE** |
| ~~Replan v2 (Population Search)~~ | replan, multi, routing | A | ○○○ | ✓ | 🔒 | XL | **v2.3 DONE** |
| Remote Execution Backends | runners, new executors | B | ○○○ | ✓ | 🔒🔒 | XL | v2.0 |

---

## Key Insights

The matrix above is still useful as a qualitative ranking of where the codebase historically saw lower or higher delivery risk, but the prose that originally followed it mixed live status with now-historical planning notes.

Current grounding for this repo review:

- Current package version is `2.4.0`.
- Latest full-suite result is about `11.3K` tests.
- Current Python source volume under `src/` is about `48.6K` LOC.
- The main branch now includes substantial Phase 2 and Phase 3 work beyond this historical snapshot: SQLite-backed Knowledge + Memory v2, policy-versioned semantic cache with negative caching, score-history persistence, multi-variant replan search, simulation-cache reuse, deterministic search benchmarks, and red-team safety-contract validation.
- Several sections below are retained as historical planning context and should not be read as the current roadmap or acceptance gate.

---

## Historical Implementation Order

### v1.24.0 — Quick wins (**SHIPPED**)
1. ~~Agent-Triggered Context Compression~~ ← `compress` signal + `compress_before`
2. ~~Event-Driven Reminders~~ ← 4 built-in triggers + custom triggers
3. ~~Watch Step Counter~~ ← `max_total_steps` + E066
4. ~~OTLP Exporter~~ ← `otel.py` + `maestro export-otel`
5. ~~Run Knowledge Expansion~~ ← 4 new patterns (cost/duration/retry/model)
6. ~~Honeypot Decoys (Kavach)~~ ← SEC019, `_inject_honeypot_decoys()`, `honeypot_triggered` event

**Actual**: +1774 tests (6726→8500), 24 modules test-expanded. All 6 features shipped.

### v1.25.0 — Production hardening
1. Staged Progressive Compaction ← runners now grade B (was C)
2. Trajectory-Level Guardrails ← scheduler now grade B
3. Security Contracts (Envelope) ← eventsource grade A
4. Dual Verification (Worktrees) ← worktree grade A
5. Multi-Dimensional Eval ← eval grade A
6. Workflow Libraries ← scaffold grade A
7. CWE Security Profiles ← runners grade B

**Estimated**: ~2000 LOC, ~100 new tests. Coverage prerequisites met (no module below grade B in primary path).

### v2.0.0 — Architectural leap
1. ~~Council Mode~~ ← **v2.1 DONE** (3 topologies: star, chain, graph)
2. ~~Replan v2~~ ← **v2.3 DONE** (multi-variant population search)
3. Remote Execution Backends [experimental]
4. A2A Engine + Server
5. ~~Python SDK~~ ← **v2.0 DONE** (29 `__all__` exports, `py.typed`)

**Estimated**: ~4000+ LOC, requires branch prototyping

---

## Historical Module Coverage Reference

Historical LOC/test ratios captured for the snapshot baseline (v1.24.0-era data):

| Module | LOC | Tests | Ratio | Grade |
|--------|-----|-------|-------|-------|
| runners | 6660 | 968 | 6.9 | B |
| loader | 2870 | 520 | 5.5 | B |
| scheduler | 2848 | 433 | 6.6 | B |
| models | 1650 | 412 | 4.0 | A |
| watch | 1354 | 280 | 4.8 | A |
| audit | 817 | 295 | 2.8 | A |
| chat | 645 | 315 | 2.0 | A |
| scaffold | 589 | 129 | 4.6 | A |
| contracts | 446 | 143 | 3.1 | A |
| suggest | 428 | 199 | 2.2 | A |
| worktree | 397 | 110 | 3.6 | A |
| live | 393 | 121 | 3.2 | A |
| eval | 385 | 113 | 3.4 | A |
| multi | 381 | 83 | 4.6 | A |
| blame | 333 | 118 | 2.8 | A |
| dynamic | 326 | 135 | 2.4 | A |
| routing | 325 | 160 | 2.0 | A |
| knowledge | 315 | 93 | 3.4 | A |
| benchmark | 271 | 54 | 5.0 | A |
| replan | 260 | 63 | 4.1 | A |
| eventsource | 227 | 72 | 3.2 | A |
| policy | 208 | 99 | 2.1 | A |

Grading scale: A ≤ 5, B ≤ 10, C ≤ 16, D > 16.

---

## How to use this matrix

Before starting any feature:

1. **Check the risk level** — Low = go. Medium = write tests first. High = prototype. Critical = defer.
2. **Check "Integration"** — ○○○ means write integration tests before shipping.
3. **Check "Coverage"** — D means the module needs edge-case L2 tests first.
4. **Check "Security"** — 🔒🔒 means security review before merge.

Update this matrix after each version release.

> Note: this file is a point-in-time historical snapshot, not a live ledger. The "update after each release" guidance reflects the original planning workflow; for current shipped status, prefer [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](../CHANGELOG.md).
