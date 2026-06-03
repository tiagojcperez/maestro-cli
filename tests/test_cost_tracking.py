from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from maestro_cli.models import (
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskSpec,
    TokenUsage,
)
from maestro_cli.runners import (
    _CostAndTokens,
    _estimate_codex_cost,
    _estimate_cost_from_tokens,
    _extract_cache_creation_tokens,
    _extract_codex_cumulative_usage,
    _extract_cost_and_tokens_from_log,
    _extract_cost_from_log,
    _load_claude_pricing_table,
    _load_codex_pricing_table,
    _load_gemini_pricing_table,
    _load_pricing_table_for_engine,
    _normalize_codex_pricing_table,
    _normalize_pricing_table,
    _resolve_model_for_pricing,
)
from maestro_cli.cost_backfill import _infer_engine


# ---------------------------------------------------------------------------
# TokenUsage dataclass
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_defaults(self) -> None:
        tu = TokenUsage()
        assert tu.input_tokens == 0
        assert tu.cached_tokens == 0
        assert tu.output_tokens == 0
        assert tu.cache_creation_tokens == 0
        assert tu.total_tokens == 0

    def test_total_tokens(self) -> None:
        tu = TokenUsage(input_tokens=100, cached_tokens=50, output_tokens=200)
        assert tu.total_tokens == 350

    def test_to_dict(self) -> None:
        tu = TokenUsage(
            input_tokens=1000, cached_tokens=500,
            output_tokens=2000, cache_creation_tokens=300,
        )
        d = tu.to_dict()
        assert d["input_tokens"] == 1000
        assert d["cached_tokens"] == 500
        assert d["output_tokens"] == 2000
        assert d["cache_creation_tokens"] == 300
        assert d["total_tokens"] == 3500

    def test_task_result_includes_token_usage(self) -> None:
        from maestro_cli.utils import now_utc
        now = now_utc()
        tu = TokenUsage(input_tokens=100, output_tokens=200)
        tr = TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="echo hi", log_path=Path("/tmp/t.log"),
            result_path=Path("/tmp/t.json"),
            cost_usd=0.01, token_usage=tu,
        )
        d = tr.to_dict()
        assert d["token_usage"]["input_tokens"] == 100
        assert d["token_usage"]["total_tokens"] == 300

    def test_task_result_token_usage_none(self) -> None:
        from maestro_cli.utils import now_utc
        now = now_utc()
        tr = TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="echo hi", log_path=Path("/tmp/t.log"),
            result_path=Path("/tmp/t.json"),
        )
        d = tr.to_dict()
        assert d["token_usage"] is None


# ---------------------------------------------------------------------------
# Pricing tables
# ---------------------------------------------------------------------------


class TestPricingTables:
    def test_codex_pricing_defaults(self) -> None:
        table = _load_codex_pricing_table()
        assert "default" in table
        inp, cached, out = table["default"]
        assert inp > 0 and out > 0

    def test_claude_pricing_defaults(self) -> None:
        table = _load_claude_pricing_table()
        assert "sonnet" in table
        assert "haiku" in table
        assert "opus" in table
        inp, cached, out = table["sonnet"]
        assert inp == 3.0
        assert out == 15.0

    def test_gemini_pricing_defaults(self) -> None:
        table = _load_gemini_pricing_table()
        assert "gemini-2.5-flash" in table
        assert "gemini-2.5-pro" in table
        inp, cached, out = table["gemini-2.5-flash"]
        assert inp == 0.30
        assert out == 2.50

    def test_load_pricing_table_for_engine(self) -> None:
        assert "sonnet" in _load_pricing_table_for_engine("claude")
        assert "default" in _load_pricing_table_for_engine("codex")
        assert "gemini-2.5-flash" in _load_pricing_table_for_engine("gemini")
        assert _load_pricing_table_for_engine("unknown") == {}

    def test_claude_pricing_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        override = json.dumps({
            "custom-model": {
                "input_per_million": 99.0,
                "cached_input_per_million": 9.0,
                "output_per_million": 199.0,
            }
        })
        monkeypatch.setenv("MAESTRO_CLAUDE_PRICING_JSON", override)
        table = _load_claude_pricing_table()
        assert "custom-model" in table
        assert table["custom-model"] == (99.0, 9.0, 199.0)
        # Defaults still present
        assert "sonnet" in table

    def test_gemini_pricing_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        override = json.dumps({
            "gemini-4-flash": {
                "input_per_million": 5.0,
                "output_per_million": 50.0,
            }
        })
        monkeypatch.setenv("MAESTRO_GEMINI_PRICING_JSON", override)
        table = _load_gemini_pricing_table()
        assert "gemini-4-flash" in table
        inp, cached, out = table["gemini-4-flash"]
        assert inp == 5.0
        assert cached == 5.0  # defaults to input when not specified
        assert out == 50.0

    def test_backward_compat_alias_normalize(self) -> None:
        """_normalize_codex_pricing_table is an alias for _normalize_pricing_table."""
        assert _normalize_codex_pricing_table is _normalize_pricing_table

    def test_backward_compat_alias_estimate(self) -> None:
        """_estimate_codex_cost is an alias for _estimate_cost_from_tokens."""
        assert _estimate_codex_cost is _estimate_cost_from_tokens


# ---------------------------------------------------------------------------
# Cost estimation from tokens
# ---------------------------------------------------------------------------


class TestEstimateCostFromTokens:
    def test_codex_estimation(self) -> None:
        pricing = _load_codex_pricing_table()
        cost = _estimate_cost_from_tokens(
            model="gpt-5.3-codex",
            input_tokens=10_000, cached_tokens=5_000, output_tokens=3_000,
            pricing=pricing,
        )
        assert cost is not None
        assert cost > 0

    def test_claude_estimation(self) -> None:
        pricing = _load_claude_pricing_table()
        cost = _estimate_cost_from_tokens(
            model="sonnet",
            input_tokens=100_000, cached_tokens=50_000, output_tokens=10_000,
            pricing=pricing,
        )
        assert cost is not None
        # sonnet: 100k/1M * 3.0 + 50k/1M * 0.30 + 10k/1M * 15.0
        expected = (100_000 / 1e6) * 3.0 + (50_000 / 1e6) * 0.30 + (10_000 / 1e6) * 15.0
        assert abs(cost - expected) < 0.001

    def test_gemini_estimation(self) -> None:
        pricing = _load_gemini_pricing_table()
        cost = _estimate_cost_from_tokens(
            model="gemini-2.5-pro",
            input_tokens=200_000, cached_tokens=100_000, output_tokens=20_000,
            pricing=pricing,
        )
        assert cost is not None
        expected = (200_000 / 1e6) * 1.25 + (100_000 / 1e6) * 0.125 + (20_000 / 1e6) * 10.0
        assert abs(cost - expected) < 0.001

    def test_unknown_model_returns_none(self) -> None:
        cost = _estimate_cost_from_tokens(
            model="nonexistent",
            input_tokens=100, cached_tokens=0, output_tokens=100,
            pricing={},
        )
        assert cost is None


# ---------------------------------------------------------------------------
# Extract cost and tokens from log
# ---------------------------------------------------------------------------


class TestExtractCostAndTokensFromLog:
    def test_claude_json_output(self, tmp_path: Path) -> None:
        """Claude CLI outputs JSON with total_cost_usd and usage."""
        log = tmp_path / "claude.log"
        payload = {
            "total_cost_usd": 0.0532,
            "usage": {
                "input_tokens": 15000,
                "cache_read_input_tokens": 8000,
                "output_tokens": 2500,
                "cache_creation_input_tokens": 1200,
            },
        }
        log.write_text(
            f"task=t1\ncommand=claude --print\n\n{json.dumps(payload)}\n\nstatus=success\n",
            encoding="utf-8",
        )
        result = _extract_cost_and_tokens_from_log(log, engine="claude", model="sonnet")
        assert result.cost_usd == 0.0532
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 15000 + 1200  # includes cache_creation
        assert result.token_usage.cached_tokens == 8000
        assert result.token_usage.output_tokens == 2500
        assert result.token_usage.cache_creation_tokens == 1200

    def test_codex_jsonl_output(self, tmp_path: Path) -> None:
        """Codex outputs JSONL with turn.completed events."""
        log = tmp_path / "codex.log"
        event1 = json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 5000, "output_tokens": 1000},
        })
        event2 = json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 3000, "output_tokens": 800},
        })
        log.write_text(
            f"task=t1\ncommand=codex exec -m gpt-5.3-codex\n\n{event1}\n{event2}\n\nstatus=success\n",
            encoding="utf-8",
        )
        result = _extract_cost_and_tokens_from_log(log, engine="codex", model="gpt-5.3-codex")
        # No direct cost, should estimate from tokens
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 8000
        assert result.token_usage.output_tokens == 1800
        # Cost estimated from pricing
        assert result.cost_usd is not None
        assert result.cost_usd > 0

    def test_gemini_json_output(self, tmp_path: Path) -> None:
        """Gemini outputs JSON with usage in stats."""
        log = tmp_path / "gemini.log"
        payload = {
            "usage": {
                "input_tokens": 20000,
                "output_tokens": 5000,
            },
        }
        log.write_text(
            f"task=t1\ncommand=gemini -m gemini-2.5-flash\n\n{json.dumps(payload)}\n\nstatus=success\n",
            encoding="utf-8",
        )
        result = _extract_cost_and_tokens_from_log(log, engine="gemini", model="flash")
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 20000
        assert result.token_usage.output_tokens == 5000
        # Cost estimated from pricing (no direct cost field)
        assert result.cost_usd is not None

    def test_no_engine_returns_cost_only(self, tmp_path: Path) -> None:
        """Without engine, still extracts cost from text patterns."""
        log = tmp_path / "task.log"
        log.write_text(
            'task=t1\ncommand=echo\n\nTotal cost: $1.23\n\nstatus=success\n',
            encoding="utf-8",
        )
        result = _extract_cost_and_tokens_from_log(log)
        assert result.cost_usd == 1.23
        assert result.token_usage is None

    def test_empty_log(self, tmp_path: Path) -> None:
        log = tmp_path / "empty.log"
        log.write_text("", encoding="utf-8")
        result = _extract_cost_and_tokens_from_log(log)
        assert result.cost_usd is None
        assert result.token_usage is None

    def test_missing_log(self, tmp_path: Path) -> None:
        log = tmp_path / "nonexistent.log"
        result = _extract_cost_and_tokens_from_log(log)
        assert result.cost_usd is None
        assert result.token_usage is None


# ---------------------------------------------------------------------------
# Backward compat: _extract_cost_from_log
# ---------------------------------------------------------------------------


class TestExtractCostFromLogBackwardCompat:
    def test_returns_cost(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        payload = {"total_cost_usd": 0.05}
        log.write_text(
            f"task=t1\n\n{json.dumps(payload)}\n\nstatus=success\n",
            encoding="utf-8",
        )
        assert _extract_cost_from_log(log) == 0.05

    def test_returns_none_when_no_cost(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("task=t1\n\nhello world\n", encoding="utf-8")
        assert _extract_cost_from_log(log) is None


# ---------------------------------------------------------------------------
# Codex cumulative usage
# ---------------------------------------------------------------------------


class TestExtractCodexCumulativeUsage:
    def test_sums_multiple_turns(self) -> None:
        lines = [
            json.dumps({"usage": {"input_tokens": 100, "output_tokens": 50}}),
            json.dumps({"usage": {"input_tokens": 200, "output_tokens": 80}}),
        ]
        result = _extract_codex_cumulative_usage(lines)
        assert result == (300, 0, 130)

    def test_no_usage_falls_back_to_byte_estimation(self) -> None:
        """Strategy 4: when no usage events exist, estimate from byte length."""
        lines = ["hello", "world"]
        result = _extract_codex_cumulative_usage(lines)
        assert result is not None, "Should fall back to byte estimation"
        input_t, cached_t, output_t = result
        assert input_t == 0, "input_tokens should be 0 (unknown)"
        assert cached_t == 0, "cached_tokens should be 0"
        assert output_t > 0, "output_tokens should be estimated from bytes"

    def test_empty_lines_returns_none(self) -> None:
        """Only empty lines should return None."""
        assert _extract_codex_cumulative_usage([]) is None
        assert _extract_codex_cumulative_usage(["", ""]) is None


# ---------------------------------------------------------------------------
# Cache creation tokens
# ---------------------------------------------------------------------------


class TestExtractCacheCreationTokens:
    def test_extracts_value(self) -> None:
        payload = json.dumps({
            "usage": {"cache_creation_input_tokens": 500},
        })
        assert _extract_cache_creation_tokens([payload]) == 500

    def test_no_value_returns_zero(self) -> None:
        assert _extract_cache_creation_tokens(["hello"]) == 0


# ---------------------------------------------------------------------------
# Resolve model for pricing
# ---------------------------------------------------------------------------


class TestResolveModelForPricing:
    def test_codex_from_command_line(self) -> None:
        lines = ["command=codex exec -m gpt-5.3-codex --approval-mode full-auto"]
        model = _resolve_model_for_pricing("codex", None, lines)
        assert model == "gpt-5.3-codex"

    def test_codex_fallback_to_task_model(self) -> None:
        lines = ["command=codex exec"]
        model = _resolve_model_for_pricing("codex", "gpt-5.2-codex", lines)
        assert model == "gpt-5.2-codex"

    def test_claude_uses_task_model(self) -> None:
        model = _resolve_model_for_pricing("claude", "sonnet", [])
        assert model == "sonnet"

    def test_gemini_resolves_alias(self) -> None:
        model = _resolve_model_for_pricing("gemini", "flash", [])
        assert model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Token aggregation in PlanRunResult
# ---------------------------------------------------------------------------


class TestPlanRunResultTokens:
    def test_total_tokens_in_to_dict(self) -> None:
        from maestro_cli.utils import now_utc
        now = now_utc()
        result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=Path("/tmp"),
            started_at=now, finished_at=now, success=True,
            total_tokens=12345,
        )
        d = result.to_dict()
        assert d["total_tokens"] == 12345

    def test_total_tokens_none(self) -> None:
        from maestro_cli.utils import now_utc
        now = now_utc()
        result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=Path("/tmp"),
            started_at=now, finished_at=now, success=True,
        )
        d = result.to_dict()
        assert d["total_tokens"] is None


# ---------------------------------------------------------------------------
# Summary includes tokens
# ---------------------------------------------------------------------------


class TestSummaryTokens:
    def _make_run_result(self, tmp_path: Path) -> tuple[PlanRunResult, PlanSpec]:
        from maestro_cli.utils import now_utc
        now = now_utc()

        tu = TokenUsage(input_tokens=5000, cached_tokens=1000, output_tokens=2000)
        tr = TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=5.0,
            command="echo hi", log_path=tmp_path / "t1.log",
            result_path=tmp_path / "t1.result.json",
            cost_usd=0.02, token_usage=tu,
        )
        (tmp_path / "t1.log").write_text("ok", encoding="utf-8")

        plan = PlanSpec(
            version=1, name="test",
            tasks=[TaskSpec(id="t1", command="echo hi")],
        )

        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=now, finished_at=now, success=True,
            task_results={"t1": tr},
            total_cost_usd=0.02, total_tokens=8000,
        )
        return run_result, plan

    def test_summary_header_has_tokens(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _write_summary
        run_result, plan = self._make_run_result(tmp_path)
        summary_path = _write_summary(run_result, plan, tmp_path)
        content = summary_path.read_text(encoding="utf-8")
        assert "| Tokens | 8,000 |" in content

    def test_summary_task_table_has_tokens(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _write_summary
        run_result, plan = self._make_run_result(tmp_path)
        summary_path = _write_summary(run_result, plan, tmp_path)
        content = summary_path.read_text(encoding="utf-8")
        assert "| Task | Status | Duration | Cost | Tokens | Engine |" in content
        assert "8,000" in content


# ---------------------------------------------------------------------------
# Backfill — engine inference
# ---------------------------------------------------------------------------


class TestBackfillInferEngine:
    def test_codex(self) -> None:
        assert _infer_engine("codex exec -m gpt-5.3-codex --approval-mode full-auto") == "codex"

    def test_claude(self) -> None:
        assert _infer_engine("claude --print -p 'hello'") == "claude"

    def test_gemini(self) -> None:
        assert _infer_engine("gemini -m gemini-2.5-flash 'do stuff'") == "gemini"

    def test_shell_command(self) -> None:
        assert _infer_engine("echo hello world") is None

    def test_empty(self) -> None:
        assert _infer_engine("") is None
