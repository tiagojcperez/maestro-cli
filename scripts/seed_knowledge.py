"""Pre-seed Knowledge Memory v2 for a plan before its first run.

Usage:
    py scripts/seed_knowledge.py

Edit the `_RECORDS` list below with your institutional knowledge and update
`_PLAN_NAME` / `_SOURCE_DIR` to match the plan you want to seed.

Background: Maestro auto-extracts knowledge after each run via
`extract_knowledge()` and auto-injects it into matching task prompts via
`{{ task_knowledge }}`. For first-time plans where you already have hard-won
insights (past incident learnings, known edge cases, "always check X" rules),
seed records before the first run so the first run benefits from them.

This script is intentionally a small template, not a CLI subcommand —
direct your edits here and rerun. See docs/PLAN_GUIDE.md for the design
rationale and an internal post-mortem.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from maestro_cli.knowledge import store_knowledge
from maestro_cli.models import KnowledgeRecord


_PLAN_NAME = "my-plan-name"  # match the `name:` field of your plan YAML
_SOURCE_DIR = Path("plans")  # directory containing the plan (or workspace_root)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_RECORDS: list[KnowledgeRecord] = [
    # Example: edge case the agent should remember to cover.
    KnowledgeRecord(
        task_id="test-target-module",
        kind="success_pattern",
        insight=(
            "Always include a regression test for the arch-import fallback "
            "path using mock.patch.dict(sys.modules, ...). Prior remediation "
            "removed dead branches; this test prevents that work from being "
            "undone."
        ),
        confidence=0.7,  # 0.5 = plausible; 0.7-0.9 = hard-won; decays over 30d
        occurrences=1,
        first_seen=_now(),
        last_seen=_now(),
    ),
    # Example: known timeout hint for a slow task.
    # KnowledgeRecord(
    #     task_id="bench-large-corpus",
    #     kind="timeout_hint",
    #     insight="Needs at least 1500s timeout on macOS — measured 1200s p95.",
    #     confidence=0.8,
    #     occurrences=1,
    #     first_seen=_now(),
    #     last_seen=_now(),
    # ),
]


def main() -> None:
    if not _RECORDS:
        print("[seed-knowledge] no records to seed — edit _RECORDS in this file")
        return
    store_knowledge(_PLAN_NAME, _SOURCE_DIR, _RECORDS)
    print(
        f"[seed-knowledge] seeded {len(_RECORDS)} record(s) for "
        f"plan='{_PLAN_NAME}' under {_SOURCE_DIR}"
    )


if __name__ == "__main__":
    main()
