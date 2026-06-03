"""Tests for cross-run budget tracking (budget.py)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from maestro_cli.budget import (
    BudgetLedgerEntry,
    _period_start,
    check_budget,
    format_budget,
    get_period_spend,
    record_cost,
)


# ---------------------------------------------------------------------------
# _period_start
# ---------------------------------------------------------------------------

class TestPeriodStart:
    def test_daily_start(self) -> None:
        now = datetime(2026, 3, 17, 14, 30, 0)
        start = _period_start("daily", now)
        assert start == datetime(2026, 3, 17, 0, 0, 0)

    def test_weekly_start_monday(self) -> None:
        # 2026-03-17 is a Tuesday
        now = datetime(2026, 3, 17, 14, 30, 0)
        start = _period_start("weekly", now)
        assert start == datetime(2026, 3, 16, 0, 0, 0)  # Monday

    def test_weekly_start_on_monday(self) -> None:
        now = datetime(2026, 3, 16, 10, 0, 0)  # Monday
        start = _period_start("weekly", now)
        assert start == datetime(2026, 3, 16, 0, 0, 0)

    def test_monthly_start(self) -> None:
        now = datetime(2026, 3, 17, 14, 30, 0)
        start = _period_start("monthly", now)
        assert start == datetime(2026, 3, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# record_cost / get_period_spend
# ---------------------------------------------------------------------------

class TestRecordAndSpend:
    def test_record_creates_ledger(self, tmp_path: Path) -> None:
        ledger = tmp_path / ".maestro-cache" / "budget_ledger.jsonl"
        record_cost(ledger, "my-plan", "run-1", 1.50)
        assert ledger.exists()
        data = json.loads(ledger.read_text(encoding="utf-8").strip())
        assert data["plan_name"] == "my-plan"
        assert data["cost_usd"] == 1.50

    def test_record_appends(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p1", "r1", 1.0)
        record_cost(ledger, "p2", "r2", 2.0)
        lines = ledger.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_record_zero_cost_ignored(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r", 0.0)
        assert not ledger.exists()

    def test_record_negative_cost_ignored(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r", -5.0)
        assert not ledger.exists()

    def test_get_period_spend_empty(self, tmp_path: Path) -> None:
        ledger = tmp_path / "nonexistent.jsonl"
        assert get_period_spend(ledger, "daily") == 0.0

    def test_get_period_spend_sums_current_period(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        now = datetime.now()
        # Write two entries: one today, one yesterday
        entries = [
            {"plan_name": "p", "run_id": "r1", "cost_usd": 3.0,
             "timestamp": now.isoformat()},
            {"plan_name": "p", "run_id": "r2", "cost_usd": 2.0,
             "timestamp": (now - timedelta(hours=1)).isoformat()},
        ]
        ledger.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        total = get_period_spend(ledger, "daily", now)
        assert total == pytest.approx(5.0)

    def test_get_period_spend_excludes_old_entries(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 17, 14, 0, 0)
        entries = [
            {"plan_name": "p", "run_id": "r1", "cost_usd": 10.0,
             "timestamp": (now - timedelta(days=2)).isoformat()},  # 2 days ago
            {"plan_name": "p", "run_id": "r2", "cost_usd": 3.0,
             "timestamp": now.isoformat()},  # today
        ]
        ledger.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        total = get_period_spend(ledger, "daily", now)
        assert total == pytest.approx(3.0)  # only today's entry

    def test_get_period_spend_weekly(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 17, 14, 0, 0)  # Tuesday
        entries = [
            {"plan_name": "p", "run_id": "r1", "cost_usd": 5.0,
             "timestamp": datetime(2026, 3, 16, 10, 0).isoformat()},  # Monday (this week)
            {"plan_name": "p", "run_id": "r2", "cost_usd": 8.0,
             "timestamp": datetime(2026, 3, 10, 10, 0).isoformat()},  # last week
        ]
        ledger.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        total = get_period_spend(ledger, "weekly", now)
        assert total == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# check_budget
# ---------------------------------------------------------------------------

class TestCheckBudget:
    def test_under_budget(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 3.0)
        allowed, spent, remaining = check_budget(ledger, "daily", 10.0)
        assert allowed is True
        assert spent == pytest.approx(3.0)
        assert remaining == pytest.approx(7.0)

    def test_over_budget(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 8.0)
        record_cost(ledger, "p", "r2", 5.0)
        allowed, spent, remaining = check_budget(ledger, "daily", 10.0)
        assert allowed is False
        assert spent == pytest.approx(13.0)
        assert remaining == pytest.approx(0.0)

    def test_empty_ledger_allowed(self, tmp_path: Path) -> None:
        ledger = tmp_path / "nonexistent.jsonl"
        allowed, spent, remaining = check_budget(ledger, "daily", 10.0)
        assert allowed is True
        assert spent == 0.0
        assert remaining == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# format_budget
# ---------------------------------------------------------------------------

class TestFormatBudget:
    def test_format_includes_periods(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        output = format_budget(ledger)
        assert "daily" in output
        assert "weekly" in output
        assert "monthly" in output

    def test_format_with_limit(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 5.0)
        output = format_budget(ledger, period="daily", max_cost_usd=10.0)
        assert "remaining" in output
        assert "$5.00" in output


# ---------------------------------------------------------------------------
# BudgetLedgerEntry
# ---------------------------------------------------------------------------

class TestBudgetLedgerEntry:
    def test_to_dict(self) -> None:
        entry = BudgetLedgerEntry(
            plan_name="test", run_id="r1",
            cost_usd=1.5, timestamp="2026-03-17T00:00:00",
        )
        d = entry.to_dict()
        assert d["plan_name"] == "test"
        assert d["cost_usd"] == 1.5


# ---------------------------------------------------------------------------
# _period_start edge cases
# ---------------------------------------------------------------------------

class TestPeriodStartEdgeCases:
    def test_monthly_last_day_of_month(self) -> None:
        """Jan 31 should start at Jan 1."""
        now = datetime(2026, 1, 31, 23, 59, 59)
        start = _period_start("monthly", now)
        assert start == datetime(2026, 1, 1, 0, 0, 0)

    def test_weekly_on_sunday(self) -> None:
        """Sunday should start at Monday of that week."""
        # 2026-03-22 is a Sunday
        now = datetime(2026, 3, 22, 18, 0, 0)
        assert now.weekday() == 6  # confirm Sunday
        start = _period_start("weekly", now)
        assert start == datetime(2026, 3, 16, 0, 0, 0)  # Monday
        assert start.weekday() == 0

    def test_daily_at_midnight(self) -> None:
        """Start == now (zeroed) when already at midnight."""
        now = datetime(2026, 6, 15, 0, 0, 0, 0)
        start = _period_start("daily", now)
        assert start == now

    def test_default_now_no_param(self) -> None:
        """Calling without now should not crash and return today's midnight for daily."""
        start = _period_start("daily")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        assert start == today

    def test_monthly_feb_28_non_leap(self) -> None:
        """Feb 28 (non-leap year) should start at Feb 1."""
        now = datetime(2027, 2, 28, 12, 0, 0)
        start = _period_start("monthly", now)
        assert start == datetime(2027, 2, 1, 0, 0, 0)

    def test_weekly_on_saturday(self) -> None:
        """Saturday should still point back to Monday."""
        # 2026-03-21 is a Saturday
        now = datetime(2026, 3, 21, 8, 0, 0)
        assert now.weekday() == 5  # confirm Saturday
        start = _period_start("weekly", now)
        assert start == datetime(2026, 3, 16, 0, 0, 0)


# ---------------------------------------------------------------------------
# record_cost edge cases
# ---------------------------------------------------------------------------

class TestRecordCostEdgeCases:
    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        """Ledger in deeply nested path should create all parents."""
        ledger = tmp_path / "a" / "b" / "c" / "d" / "ledger.jsonl"
        record_cost(ledger, "deep-plan", "r1", 2.0)
        assert ledger.exists()
        data = json.loads(ledger.read_text(encoding="utf-8").strip())
        assert data["plan_name"] == "deep-plan"
        assert data["cost_usd"] == 2.0

    def test_timestamp_is_iso_format(self, tmp_path: Path) -> None:
        """Verify recorded timestamp is valid ISO format."""
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 1.0)
        data = json.loads(ledger.read_text(encoding="utf-8").strip())
        ts = data["timestamp"]
        # Should be parseable as ISO datetime
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)

    def test_multiple_plans_in_same_ledger(self, tmp_path: Path) -> None:
        """Different plan_name values coexist in the same ledger."""
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "plan-alpha", "r1", 1.0)
        record_cost(ledger, "plan-beta", "r2", 2.0)
        record_cost(ledger, "plan-gamma", "r3", 3.0)
        lines = ledger.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        plans = {json.loads(line)["plan_name"] for line in lines}
        assert plans == {"plan-alpha", "plan-beta", "plan-gamma"}

    def test_cost_precision_preserved(self, tmp_path: Path) -> None:
        """Float precision should be preserved in the ledger."""
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 1.234567)
        data = json.loads(ledger.read_text(encoding="utf-8").strip())
        assert data["cost_usd"] == pytest.approx(1.234567)


# ---------------------------------------------------------------------------
# get_period_spend edge cases
# ---------------------------------------------------------------------------

class TestGetPeriodSpendEdgeCases:
    def test_corrupt_json_lines_skipped(self, tmp_path: Path) -> None:
        """Corrupt JSON should be skipped, valid lines still summed."""
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 17, 14, 0, 0)
        lines = [
            '{"plan_name":"p","run_id":"r1","cost_usd":5.0,"timestamp":"2026-03-17T10:00:00"}',
            "this is not json {{{",
            '{"plan_name":"p","run_id":"r2","cost_usd":3.0,"timestamp":"2026-03-17T12:00:00"}',
        ]
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        total = get_period_spend(ledger, "daily", now)
        assert total == pytest.approx(8.0)

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        """Empty lines and whitespace-only lines should be skipped."""
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 17, 14, 0, 0)
        lines = [
            "",
            '{"plan_name":"p","run_id":"r1","cost_usd":4.0,"timestamp":"2026-03-17T10:00:00"}',
            "   ",
            "",
            '{"plan_name":"p","run_id":"r2","cost_usd":2.0,"timestamp":"2026-03-17T12:00:00"}',
        ]
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        total = get_period_spend(ledger, "daily", now)
        assert total == pytest.approx(6.0)

    def test_missing_timestamp_field_skipped(self, tmp_path: Path) -> None:
        """Entry without 'timestamp' should be skipped."""
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 17, 14, 0, 0)
        lines = [
            '{"plan_name":"p","run_id":"r1","cost_usd":5.0}',
            '{"plan_name":"p","run_id":"r2","cost_usd":3.0,"timestamp":"2026-03-17T12:00:00"}',
        ]
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        total = get_period_spend(ledger, "daily", now)
        # First entry has empty timestamp -> fromisoformat("") raises ValueError -> skipped
        assert total == pytest.approx(3.0)

    def test_missing_cost_usd_field_treated_as_zero(self, tmp_path: Path) -> None:
        """Entry without 'cost_usd' should contribute 0."""
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 17, 14, 0, 0)
        lines = [
            '{"plan_name":"p","run_id":"r1","timestamp":"2026-03-17T10:00:00"}',
            '{"plan_name":"p","run_id":"r2","cost_usd":7.0,"timestamp":"2026-03-17T12:00:00"}',
        ]
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        total = get_period_spend(ledger, "daily", now)
        assert total == pytest.approx(7.0)

    def test_monthly_period_span_cross_boundary(self, tmp_path: Path) -> None:
        """Only entries from the current month are summed."""
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 5, 10, 0, 0)  # March 5
        entries = [
            {"plan_name": "p", "run_id": "r1", "cost_usd": 10.0,
             "timestamp": "2026-02-28T23:59:59"},  # Feb (last month)
            {"plan_name": "p", "run_id": "r2", "cost_usd": 4.0,
             "timestamp": "2026-03-01T00:00:01"},  # March 1 (this month)
            {"plan_name": "p", "run_id": "r3", "cost_usd": 6.0,
             "timestamp": "2026-03-05T09:00:00"},  # March 5 (this month)
        ]
        ledger.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        total = get_period_spend(ledger, "monthly", now)
        assert total == pytest.approx(10.0)  # only March entries

    def test_oserror_on_read_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OSError during file read should return 0.0."""
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text(
            '{"plan_name":"p","run_id":"r1","cost_usd":5.0,'
            '"timestamp":"2026-03-17T10:00:00"}\n',
            encoding="utf-8",
        )

        def raise_oserror(*args: object, **kwargs: object) -> str:
            raise OSError("disk failure")

        monkeypatch.setattr(Path, "read_text", raise_oserror)
        total = get_period_spend(ledger, "daily")
        assert total == 0.0

    def test_all_entries_outside_period(self, tmp_path: Path) -> None:
        """All entries before the current period returns 0.0."""
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 17, 14, 0, 0)
        entries = [
            {"plan_name": "p", "run_id": "r1", "cost_usd": 10.0,
             "timestamp": "2026-03-15T10:00:00"},
            {"plan_name": "p", "run_id": "r2", "cost_usd": 20.0,
             "timestamp": "2026-03-14T10:00:00"},
        ]
        ledger.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        total = get_period_spend(ledger, "daily", now)
        assert total == pytest.approx(0.0)

    def test_invalid_timestamp_format_skipped(self, tmp_path: Path) -> None:
        """Entry with unparseable timestamp should be skipped."""
        ledger = tmp_path / "ledger.jsonl"
        now = datetime(2026, 3, 17, 14, 0, 0)
        lines = [
            '{"plan_name":"p","run_id":"r1","cost_usd":5.0,"timestamp":"not-a-date"}',
            '{"plan_name":"p","run_id":"r2","cost_usd":3.0,"timestamp":"2026-03-17T12:00:00"}',
        ]
        ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
        total = get_period_spend(ledger, "daily", now)
        assert total == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# check_budget edge cases
# ---------------------------------------------------------------------------

class TestCheckBudgetEdgeCases:
    def test_exactly_at_budget_not_allowed(self, tmp_path: Path) -> None:
        """spent == max_cost_usd should NOT be allowed (requires spent < max)."""
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 10.0)
        allowed, spent, remaining = check_budget(ledger, "daily", 10.0)
        assert allowed is False
        assert spent == pytest.approx(10.0)
        assert remaining == pytest.approx(0.0)

    def test_zero_budget_nothing_allowed(self, tmp_path: Path) -> None:
        """With max_cost_usd=0, even 0 spend is not allowed (0 < 0 is False)."""
        ledger = tmp_path / "nonexistent.jsonl"
        allowed, spent, remaining = check_budget(ledger, "daily", 0.0)
        assert allowed is False
        assert spent == pytest.approx(0.0)
        assert remaining == pytest.approx(0.0)

    def test_very_small_remaining(self, tmp_path: Path) -> None:
        """$0.01 remaining should still be allowed."""
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 9.99)
        allowed, spent, remaining = check_budget(ledger, "daily", 10.0)
        assert allowed is True
        assert spent == pytest.approx(9.99)
        assert remaining == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# format_budget edge cases
# ---------------------------------------------------------------------------

class TestFormatBudgetEdgeCases:
    def test_no_entries_all_zero(self, tmp_path: Path) -> None:
        """All periods show $0.00 when ledger doesn't exist."""
        ledger = tmp_path / "nonexistent.jsonl"
        output = format_budget(ledger)
        assert "$0.00" in output
        assert "daily" in output
        assert "weekly" in output
        assert "monthly" in output

    def test_with_period_but_no_max(self, tmp_path: Path) -> None:
        """Passing period without max_cost_usd shows just periods (no remaining)."""
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 5.0)
        output = format_budget(ledger, period="daily")
        # Without max_cost_usd, header is generic (no percentage)
        assert "[maestro] budget:" in output
        assert "remaining" not in output
        assert "daily" in output

    def test_with_max_cost_zero_no_div_by_zero(self, tmp_path: Path) -> None:
        """max_cost_usd=0 should not cause division by zero; percentage should be 0."""
        ledger = tmp_path / "nonexistent.jsonl"
        # max_cost_usd=0 is falsy, so the `if period and max_cost_usd:` branch
        # is not entered — verify it doesn't crash
        output = format_budget(ledger, period="daily", max_cost_usd=0.0)
        assert "[maestro] budget:" in output

    def test_large_numbers_formatted(self, tmp_path: Path) -> None:
        """Large dollar amounts should be formatted properly."""
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 1234.56)
        output = format_budget(ledger, period="daily", max_cost_usd=5000.0)
        assert "$1234.56" in output
        assert "$5000.00" in output
        assert "remaining" in output

    def test_format_percentage_display(self, tmp_path: Path) -> None:
        """Verify percentage calculation in header."""
        ledger = tmp_path / "ledger.jsonl"
        record_cost(ledger, "p", "r1", 2.50)
        output = format_budget(ledger, period="daily", max_cost_usd=10.0)
        assert "(25%)" in output


# ---------------------------------------------------------------------------
# BudgetLedgerEntry edge cases
# ---------------------------------------------------------------------------

class TestBudgetLedgerEntryEdgeCases:
    def test_roundtrip_to_dict_json_parse(self) -> None:
        """to_dict -> json.dumps -> json.loads produces identical values."""
        entry = BudgetLedgerEntry(
            plan_name="roundtrip-plan",
            run_id="run-42",
            cost_usd=3.14159,
            timestamp="2026-03-17T14:30:00",
        )
        d = entry.to_dict()
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["plan_name"] == entry.plan_name
        assert parsed["run_id"] == entry.run_id
        assert parsed["cost_usd"] == pytest.approx(entry.cost_usd)
        assert parsed["timestamp"] == entry.timestamp

    def test_budget_periods_constant(self) -> None:
        """BUDGET_PERIODS should contain daily, weekly, monthly."""
        from maestro_cli.budget import BUDGET_PERIODS
        assert "daily" in BUDGET_PERIODS
        assert "weekly" in BUDGET_PERIODS
        assert "monthly" in BUDGET_PERIODS
        assert len(BUDGET_PERIODS) == 3
