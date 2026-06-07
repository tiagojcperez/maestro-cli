"""Cross-run budget tracking with daily/weekly/monthly caps.

Budget ledger is stored as `.maestro-cache/budget_ledger.jsonl`.
Each line records the cost of a completed plan run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

BudgetPeriod = Literal["daily", "weekly", "monthly"]
BUDGET_PERIODS: set[str] = {"daily", "weekly", "monthly"}

_DEFAULT_LEDGER_PATH = Path(".maestro-cache") / "budget_ledger.jsonl"


@dataclass
class BudgetLedgerEntry:
    """A single cost record in the budget ledger."""

    plan_name: str
    run_id: str
    cost_usd: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "run_id": self.run_id,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
        }


def _period_start(period: BudgetPeriod, now: datetime | None = None) -> datetime:
    """Compute the start of the current budget period."""
    now = now or datetime.now()
    if period == "daily":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "weekly":
        # Monday at midnight
        start = now - timedelta(days=now.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0)
    # monthly
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def record_cost(
    ledger_path: Path,
    plan_name: str,
    run_id: str,
    cost_usd: float,
) -> None:
    """Append a cost entry to the budget ledger."""
    if cost_usd <= 0:
        return
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    entry = BudgetLedgerEntry(
        plan_name=plan_name,
        run_id=run_id,
        cost_usd=cost_usd,
        timestamp=datetime.now().isoformat(),
    )
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_dict()) + "\n")
        fh.flush()


def get_period_spend(
    ledger_path: Path,
    period: BudgetPeriod,
    now: datetime | None = None,
) -> float:
    """Sum costs from the current budget period."""
    if not ledger_path.exists():
        return 0.0
    start = _period_start(period, now)
    total = 0.0
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ts_str = data.get("timestamp", "")
                ts = datetime.fromisoformat(ts_str)
                if ts >= start:
                    total += float(data.get("cost_usd", 0))
            except (ValueError, TypeError):
                continue
    except OSError:
        return 0.0
    return total


def check_budget(
    ledger_path: Path,
    period: BudgetPeriod,
    max_cost_usd: float,
) -> tuple[bool, float, float]:
    """Check if the budget period has been exceeded.

    Returns (allowed, spent, remaining).
    """
    spent = get_period_spend(ledger_path, period)
    remaining = max(0.0, max_cost_usd - spent)
    return spent < max_cost_usd, spent, remaining


def format_budget(
    ledger_path: Path,
    period: BudgetPeriod | None = None,
    max_cost_usd: float | None = None,
) -> str:
    """Format budget status for CLI display."""
    lines: list[str] = []
    for p in ["daily", "weekly", "monthly"]:
        spent = get_period_spend(ledger_path, p)  # type: ignore[arg-type]
        lines.append(f"  {p}: ${spent:.2f}")
    header = "[maestro] budget:"
    if period and max_cost_usd:
        spent = get_period_spend(ledger_path, period)
        remaining = max(0.0, max_cost_usd - spent)
        pct = (spent / max_cost_usd * 100) if max_cost_usd > 0 else 0
        header = f"[maestro] budget ({period}): ${spent:.2f} / ${max_cost_usd:.2f} ({pct:.0f}%)"
        lines.insert(0, f"  remaining: ${remaining:.2f}")
    return header + "\n" + "\n".join(lines)
