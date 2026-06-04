from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.models import TokenUsage
from maestro_cli.plugins import CostExtraction, EnginePlugin, PluginResolutionError
from maestro_cli.runners import (
    _extract_cost_and_tokens_from_log,
    _extract_cost_from_log,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plugin(
    *,
    name: str = "custom",
    extract_cost: object | None = None,
) -> EnginePlugin:
    """Build a minimal EnginePlugin; build_command is never invoked here."""
    return EnginePlugin(
        name=name,
        build_command=lambda ctx: ([name], False),
        extract_cost=extract_cost,  # type: ignore[arg-type]
    )


def _patch_plugin_lookup(
    monkeypatch: pytest.MonkeyPatch,
    plugin: EnginePlugin | None,
) -> None:
    """Patch runners.get_engine_plugin so the registered plugin is deterministic.

    When ``plugin`` is None, the lookup raises PluginResolutionError, which
    _get_registered_engine_plugin swallows -> returns None (text-based path).
    """

    def _lookup(_name: str) -> EnginePlugin:
        if plugin is None:
            raise PluginResolutionError("no plugin")
        return plugin

    monkeypatch.setattr("maestro_cli.runners.get_engine_plugin", _lookup)
    # Avoid touching the real builtin registry loader during lookup.
    monkeypatch.setattr(
        "maestro_cli.runners._ensure_builtin_engine_plugins_registered",
        lambda: None,
    )


# ===========================================================================
# Plugin extract_cost path (runners.py L4290-4304)
# ===========================================================================


class TestPluginExtractCostPath:
    """Drive the ``plugin.extract_cost`` branch in _extract_cost_and_tokens_from_log."""

    def test_extract_cost_with_full_token_usage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plugin returns cost + full token info -> TokenUsage built (L4296-4304)."""
        log = tmp_path / "task.log"
        log.write_text("irrelevant body\n", encoding="utf-8")

        captured: dict[str, object] = {}

        def _extract(path: Path, model: str | None) -> CostExtraction:
            captured["path"] = path
            captured["model"] = model
            return CostExtraction(
                cost_usd=1.25,
                input_tokens=100,
                cached_tokens=10,
                output_tokens=40,
                cache_creation_tokens=15,
            )

        plugin = _make_plugin(extract_cost=_extract)
        _patch_plugin_lookup(monkeypatch, plugin)

        result = _extract_cost_and_tokens_from_log(log, engine="custom", model="m1")

        # Plugin was consulted with the log path and model.
        assert captured["path"] == log
        assert captured["model"] == "m1"

        assert result.cost_usd == pytest.approx(1.25)
        assert isinstance(result.token_usage, TokenUsage)
        # input_tokens = extracted.input_tokens + max(cache_creation, 0) = 100 + 15
        assert result.token_usage.input_tokens == 115
        assert result.token_usage.cached_tokens == 10
        assert result.token_usage.output_tokens == 40
        assert result.token_usage.cache_creation_tokens == 15

    def test_extract_cost_negative_cache_creation_clamped_to_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Negative cache_creation_tokens is clamped via max(..., 0) (L4299, 4302)."""
        log = tmp_path / "task.log"
        log.write_text("body\n", encoding="utf-8")

        def _extract(_path: Path, _model: str | None) -> CostExtraction:
            return CostExtraction(
                cost_usd=0.5,
                input_tokens=200,
                cached_tokens=0,
                output_tokens=80,
                cache_creation_tokens=-5,
            )

        plugin = _make_plugin(extract_cost=_extract)
        _patch_plugin_lookup(monkeypatch, plugin)

        result = _extract_cost_and_tokens_from_log(log, engine="custom")

        assert result.cost_usd == pytest.approx(0.5)
        assert isinstance(result.token_usage, TokenUsage)
        # cache_creation clamped to 0; input_tokens = 200 + 0
        assert result.token_usage.input_tokens == 200
        assert result.token_usage.cache_creation_tokens == 0

    def test_extract_cost_cost_only_no_token_usage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plugin returns cost but missing token counts -> no TokenUsage, early return (L4304)."""
        log = tmp_path / "task.log"
        log.write_text("body\n", encoding="utf-8")

        def _extract(_path: Path, _model: str | None) -> CostExtraction:
            # input_tokens is None -> the TokenUsage block is skipped.
            return CostExtraction(cost_usd=3.0, input_tokens=None, output_tokens=None)

        plugin = _make_plugin(extract_cost=_extract)
        _patch_plugin_lookup(monkeypatch, plugin)

        result = _extract_cost_and_tokens_from_log(log, engine="custom")

        assert result.cost_usd == pytest.approx(3.0)
        assert result.token_usage is None

    def test_extract_cost_output_tokens_none_skips_token_usage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """input present but output None -> token block skipped, still returns (L4297, 4304)."""
        log = tmp_path / "task.log"
        log.write_text("body\n", encoding="utf-8")

        def _extract(_path: Path, _model: str | None) -> CostExtraction:
            return CostExtraction(cost_usd=0.9, input_tokens=50, output_tokens=None)

        plugin = _make_plugin(extract_cost=_extract)
        _patch_plugin_lookup(monkeypatch, plugin)

        result = _extract_cost_and_tokens_from_log(log, engine="custom")

        assert result.cost_usd == pytest.approx(0.9)
        assert result.token_usage is None

    def test_extract_cost_raises_is_swallowed_and_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plugin.extract_cost raising is caught (L4293-4294); falls through to text path."""
        # Put a directly-extractable cost line in the log so the fallthrough path
        # yields a deterministic, non-None cost (proving we did NOT early-return).
        log = tmp_path / "task.log"
        log.write_text('{"total_cost_usd": 7.5}\n', encoding="utf-8")

        def _extract(_path: Path, _model: str | None) -> CostExtraction:
            raise RuntimeError("boom")

        plugin = _make_plugin(extract_cost=_extract)
        _patch_plugin_lookup(monkeypatch, plugin)

        result = _extract_cost_and_tokens_from_log(log, engine="custom")

        # extracted is None -> no early return; text-based extraction kicks in.
        assert result.cost_usd == pytest.approx(7.5)

    def test_extract_cost_returns_none_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plugin returns None -> ``if extracted is not None`` is False, no early return."""
        log = tmp_path / "task.log"
        log.write_text('{"costUSD": 2.0}\n', encoding="utf-8")

        def _extract(_path: Path, _model: str | None) -> None:
            return None

        plugin = _make_plugin(extract_cost=_extract)
        _patch_plugin_lookup(monkeypatch, plugin)

        result = _extract_cost_and_tokens_from_log(log, engine="custom")

        assert result.cost_usd == pytest.approx(2.0)


# ===========================================================================
# ollama / llama short-circuit (zero cost) — guards against regressions in the
# branch that precedes the plugin lookup.
# ===========================================================================


class TestLocalEngineZeroCost:
    @pytest.mark.parametrize("engine", ["ollama", "llama"])
    def test_local_engine_returns_zero_cost(
        self, tmp_path: Path, engine: str
    ) -> None:
        log = tmp_path / "task.log"
        log.write_text("anything\n", encoding="utf-8")
        result = _extract_cost_and_tokens_from_log(log, engine=engine)
        assert result.cost_usd == 0.0
        assert result.token_usage is None


# ===========================================================================
# Qwen token-usage branch (runners.py L4325-4330)
# ===========================================================================


class TestQwenUsageBranch:
    """Drive the dedicated ``engine == 'qwen'`` token-extraction loop."""

    def test_qwen_usage_line_parsed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A qwen log with a JSON usage line populates token_usage via the qwen loop."""
        # No plugin -> falls through to the text/usage path; engine == "qwen"
        # selects the dedicated reversed-tail loop (L4327-4330).
        _patch_plugin_lookup(monkeypatch, None)

        log = tmp_path / "qwen.log"
        log.write_text(
            "command=qwen-code --prompt hi\n"
            "some intermediate output\n"
            '{"usage": {"input_tokens": 120, "output_tokens": 45, '
            '"cached_input_tokens": 12}}\n',
            encoding="utf-8",
        )

        result = _extract_cost_and_tokens_from_log(log, engine="qwen", model="coder")

        assert isinstance(result.token_usage, TokenUsage)
        assert result.token_usage.input_tokens == 120
        assert result.token_usage.output_tokens == 45
        assert result.token_usage.cached_tokens == 12

    def test_qwen_no_usage_line_leaves_tokens_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Qwen loop runs but finds no usage -> token_usage stays None."""
        _patch_plugin_lookup(monkeypatch, None)

        log = tmp_path / "qwen.log"
        log.write_text(
            "command=qwen-code --prompt hi\n"
            "plain text with no json usage at all\n",
            encoding="utf-8",
        )

        result = _extract_cost_and_tokens_from_log(log, engine="qwen")
        assert result.token_usage is None


# ===========================================================================
# _extract_cost_from_log engine detection from command header
# (runners.py L4376-4393, specifically L4382/4384/4386/4388/4390/4392)
# ===========================================================================


class TestExtractCostFromLogEngineDetection:
    """Each ``command=`` header maps to a specific engine branch.

    We patch the plugin lookup to None so detection routes to the deterministic
    text-based cost path, and embed a directly-extractable cost so the call
    returns a stable, non-None float for the recognised engines.
    """

    def _write_log(self, tmp_path: Path, command_line: str) -> Path:
        log = tmp_path / "task.log"
        log.write_text(
            f"{command_line}\n"
            "intermediate output line\n"
            '{"total_cost_usd": 4.2}\n',
            encoding="utf-8",
        )
        return log

    def test_detects_claude(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_plugin_lookup(monkeypatch, None)
        log = self._write_log(tmp_path, "command=claude --print --model sonnet")
        assert _extract_cost_from_log(log) == pytest.approx(4.2)

    def test_detects_gemini(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_plugin_lookup(monkeypatch, None)
        log = self._write_log(tmp_path, "command=gemini -m flash")
        assert _extract_cost_from_log(log) == pytest.approx(4.2)

    def test_detects_copilot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_plugin_lookup(monkeypatch, None)
        log = self._write_log(tmp_path, "command=copilot --autopilot --model sonnet")
        assert _extract_cost_from_log(log) == pytest.approx(4.2)

    def test_detects_qwen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Note: detection matches the literal substring "qwen-code".
        _patch_plugin_lookup(monkeypatch, None)
        log = self._write_log(tmp_path, "command=qwen-code --model coder")
        assert _extract_cost_from_log(log) == pytest.approx(4.2)

    def test_detects_ollama_returns_zero_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ollama short-circuits to cost 0.0 regardless of any cost line.
        _patch_plugin_lookup(monkeypatch, None)
        log = self._write_log(tmp_path, "command=ollama run llama3")
        assert _extract_cost_from_log(log) == 0.0

    def test_detects_llama_returns_zero_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # llama-cli short-circuits to cost 0.0 (local engine).
        _patch_plugin_lookup(monkeypatch, None)
        log = self._write_log(tmp_path, "command=llama-cli -m llama3 -p hi")
        assert _extract_cost_from_log(log) == 0.0

    def test_unrecognised_command_leaves_engine_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A command header with no known engine keyword -> engine stays None."""
        _patch_plugin_lookup(monkeypatch, None)
        log = tmp_path / "task.log"
        log.write_text(
            "command=mystery-tool --do-thing\n"
            '{"total_cost_usd": 1.1}\n',
            encoding="utf-8",
        )
        # Still extracts the direct cost; engine inference simply found nothing.
        assert _extract_cost_from_log(log) == pytest.approx(1.1)

    def test_no_command_header_returns_direct_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_plugin_lookup(monkeypatch, None)
        log = tmp_path / "task.log"
        log.write_text('{"cost_usd": 0.33}\n', encoding="utf-8")
        assert _extract_cost_from_log(log) == pytest.approx(0.33)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """OSError on read -> None (covers the early-return guard)."""
        missing = tmp_path / "does-not-exist.log"
        assert _extract_cost_from_log(missing) is None
