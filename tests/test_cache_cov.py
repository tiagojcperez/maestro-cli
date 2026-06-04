from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from maestro_cli.cache import (
    _effective_negative_cache_ttl_sec,
    _parse_cache_timestamp,
    _resolve_model_for_engine,
    _stem_model_family,
    cache_clear,
    cache_lookup,
    cache_stats,
    cache_store,
    compute_plan_hash,
    compute_simulation_plan_hash,
    model_family_for_engine,
)
from maestro_cli.models import (
    PlanDefaults,
    PlanSpec,
    TaskResult,
    TaskSpec,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_plan(
    tasks: list[TaskSpec],
    tmp_path: Path,
    *,
    defaults: PlanDefaults | None = None,
) -> PlanSpec:
    source_path = tmp_path / "plan.yaml"
    source_path.write_text("version: 1\nname: test\n", encoding="utf-8")
    return PlanSpec(
        version=1,
        name="test-plan",
        defaults=defaults or PlanDefaults(),
        tasks=tasks,
        source_path=source_path,
    )


def _make_result(
    tmp_path: Path,
    *,
    task_id: str = "t1",
    status: str = "success",
    exit_code: int = 0,
    message: str = "",
) -> TaskResult:
    now = datetime.now(timezone.utc)
    return TaskResult(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        exit_code=exit_code,
        started_at=now,
        finished_at=now,
        duration_sec=1.0,
        command="echo ok",
        log_path=tmp_path / f"{task_id}.log",
        result_path=tmp_path / f"{task_id}.result.json",
        message=message,
        stdout_tail="done",
        cost_usd=0.01,
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


# ---------------------------------------------------------------------------
# _resolve_model_for_engine — all engine branches
# ---------------------------------------------------------------------------


class TestResolveModelForEngine:
    def test_gemini_alias_resolved(self) -> None:
        assert _resolve_model_for_engine("gemini", "flash") == "gemini-2.5-flash"

    def test_copilot_alias_resolved(self) -> None:
        assert _resolve_model_for_engine("copilot", "sonnet") == "claude-sonnet-4.6"

    def test_qwen_alias_resolved(self) -> None:
        assert _resolve_model_for_engine("qwen", "coder") == "qwen-coder-plus"

    def test_ollama_alias_resolved(self) -> None:
        assert _resolve_model_for_engine("ollama", "llama3") == "llama3"

    def test_llama_alias_resolved(self) -> None:
        assert _resolve_model_for_engine("llama", "llama3") == "llama-3-8b"

    def test_codex_alias_resolved(self) -> None:
        assert _resolve_model_for_engine("codex", "5.4") == "gpt-5.4-codex"

    def test_claude_alias_resolved(self) -> None:
        assert _resolve_model_for_engine("claude", "sonnet") == "sonnet"

    def test_unknown_engine_passes_model_through(self) -> None:
        # — fallback `return model` for an engine not in the dispatch.
        assert _resolve_model_for_engine("mystery", "some-model") == "some-model"

    def test_unknown_engine_with_none_model(self) -> None:
        assert _resolve_model_for_engine("mystery", None) is None


# ---------------------------------------------------------------------------
# _stem_model_family
# ---------------------------------------------------------------------------


class TestStemModelFamily:
    def test_strips_tag_after_colon(self) -> None:
        # "llama3:latest" -> drop ":latest", then trailing digits.
        assert _stem_model_family("llama3:latest") == "llama"

    def test_strips_namespace_before_slash(self) -> None:
        assert _stem_model_family("library/codellama") == "codellama"

    def test_strips_trailing_size_suffix(self) -> None:
        # Only the trailing numeric+unit token is stripped (e.g. "-8b").
        assert _stem_model_family("llama-3-8b") == "llama-3"

    def test_non_alnum_collapsed_to_hyphen(self) -> None:
        assert _stem_model_family("deepseek_coder") == "deepseek-coder"

    def test_all_digits_falls_back_to_lowered_model(self) -> None:
        # Stemming a pure-numeric token strips everything, so the `or` fallback
        # branch on returns the lowered original.
        result = _stem_model_family("123")
        assert result == "123"

    def test_whitespace_and_case_normalized(self) -> None:
        assert _stem_model_family("  MISTRAL  ") == "mistral"


# ---------------------------------------------------------------------------
# model_family_for_engine
# ---------------------------------------------------------------------------


class TestModelFamilyForEngine:
    def test_none_resolved_model_returns_none(self) -> None:
        assert model_family_for_engine("claude", None) is None

    def test_empty_normalized_returns_none(self) -> None:
        # — resolved model is whitespace-only -> normalized empty.
        assert model_family_for_engine("gemini", "   ") is None

    def test_native_engines_return_engine_name(self) -> None:
        for engine in ("codex", "claude", "gemini", "qwen"):
            assert model_family_for_engine(engine, "anything") == engine

    def test_copilot_anthropic_models(self) -> None:
        # — claude/haiku/sonnet/opus tokens -> "anthropic".
        assert model_family_for_engine("copilot", "claude-opus-4.6") == "anthropic"
        assert model_family_for_engine("copilot", "sonnet") == "anthropic"

    def test_copilot_gemini_model(self) -> None:
        # .
        assert model_family_for_engine("copilot", "gemini-pro") == "gemini"

    def test_copilot_grok_model(self) -> None:
        # — grok is not an alias so it passes through.
        assert model_family_for_engine("copilot", "grok-code-fast-1") == "grok"

    def test_copilot_openai_model(self) -> None:
        # — gpt/codex/o-series tokens -> "openai".
        assert model_family_for_engine("copilot", "gpt-5.4-codex") == "openai"

    def test_copilot_unmatched_returns_copilot(self) -> None:
        # — none of the token sets match.
        assert model_family_for_engine("copilot", "mistral-large") == "copilot"

    def test_ollama_stems_family(self) -> None:
        # — ollama delegates to _stem_model_family.
        assert model_family_for_engine("ollama", "llama3") == "llama"

    def test_llama_stems_family(self) -> None:
        # — llama delegates to _stem_model_family.
        assert model_family_for_engine("llama", "codellama") == "codellama"

    def test_unknown_engine_returns_normalized(self) -> None:
        # — engine not in any branch returns the normalized model.
        assert model_family_for_engine("mystery", "Custom-Model") == "custom-model"


# ---------------------------------------------------------------------------
# _effective_negative_cache_ttl_sec
# ---------------------------------------------------------------------------


class TestEffectiveNegativeCacheTtl:
    def test_explicit_ttl_clamped_to_non_negative(self) -> None:
        # — explicit ttl provided, clamped via max(0, ...).
        assert _effective_negative_cache_ttl_sec(None, explicit_ttl_sec=42) == 42

    def test_explicit_negative_ttl_clamped_to_zero(self) -> None:
        assert _effective_negative_cache_ttl_sec(None, explicit_ttl_sec=-5) == 0

    def test_task_ttl_used_when_no_explicit(self) -> None:
        task = TaskSpec(id="t", command="echo ok", negative_cache_ttl_sec=99)
        assert _effective_negative_cache_ttl_sec(task) == 99

    def test_default_used_when_nothing_set(self) -> None:
        assert _effective_negative_cache_ttl_sec(None) == 300


# ---------------------------------------------------------------------------
# _parse_cache_timestamp
# ---------------------------------------------------------------------------


class TestParseCacheTimestamp:
    def test_non_string_returns_none(self) -> None:
        # — not a str.
        assert _parse_cache_timestamp(12345) is None

    def test_empty_string_returns_none(self) -> None:
        # — empty string.
        assert _parse_cache_timestamp("") is None

    def test_invalid_iso_returns_none(self) -> None:
        # — ValueError from fromisoformat.
        assert _parse_cache_timestamp("not-a-timestamp") is None

    def test_naive_timestamp_gets_utc_tzinfo(self) -> None:
        # — naive datetime gets utc tzinfo attached.
        parsed = _parse_cache_timestamp("2026-01-01T00:00:00")
        assert parsed is not None
        assert parsed.tzinfo is timezone.utc

    def test_aware_timestamp_converted_to_utc(self) -> None:
        parsed = _parse_cache_timestamp("2026-01-01T05:00:00+05:00")
        assert parsed is not None
        assert parsed.hour == 0
        assert parsed.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# compute_plan_hash — cycle / unknown task
# ---------------------------------------------------------------------------


class TestComputePlanHashErrors:
    def test_dependency_cycle_raises(self, tmp_path: Path) -> None:
        # — mutual dependency triggers cycle detection.
        task_a = TaskSpec(id="a", command="echo a", depends_on=["b"])
        task_b = TaskSpec(id="b", command="echo b", depends_on=["a"])
        plan = _build_plan([task_a, task_b], tmp_path)
        with pytest.raises(ValueError, match="cycle"):
            compute_plan_hash(plan)

    def test_unknown_dependency_raises(self, tmp_path: Path) -> None:
        # — task depends on a non-existent task id.
        task = TaskSpec(id="a", command="echo a", depends_on=["ghost"])
        plan = _build_plan([task], tmp_path)
        with pytest.raises(ValueError, match="Unknown task"):
            compute_plan_hash(plan)


# ---------------------------------------------------------------------------
# compute_simulation_plan_hash — cycle / unknown task
# ---------------------------------------------------------------------------


class TestComputeSimulationPlanHashErrors:
    def test_dependency_cycle_raises(self, tmp_path: Path) -> None:
        # — mutual dependency triggers cycle detection.
        task_a = TaskSpec(id="a", command="echo a", depends_on=["b"])
        task_b = TaskSpec(id="b", command="echo b", depends_on=["a"])
        plan = _build_plan([task_a, task_b], tmp_path)
        with pytest.raises(ValueError, match="cycle"):
            compute_simulation_plan_hash(plan)

    def test_unknown_dependency_raises(self, tmp_path: Path) -> None:
        # — task depends on a non-existent task id.
        task = TaskSpec(id="a", command="echo a", depends_on=["ghost"])
        plan = _build_plan([task], tmp_path)
        with pytest.raises(ValueError, match="Unknown task"):
            compute_simulation_plan_hash(plan)


# ---------------------------------------------------------------------------
# cache_lookup — rmtree OSError on expired negative entry
# ---------------------------------------------------------------------------


class TestCacheLookupRmtreeError:
    def test_expired_negative_entry_rmtree_oserror_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "cache"
        task_hash = "ab" + "c" * 62
        entry_dir = cache_dir / task_hash[:2] / task_hash
        entry_dir.mkdir(parents=True)
        expired = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        (entry_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "_cache_kind": "negative",
                    "_cache_expires_at": expired,
                }
            ),
            encoding="utf-8",
        )

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("cannot remove")

        monkeypatch.setattr("maestro_cli.cache.shutil.rmtree", _boom)

        # — OSError from rmtree is caught and lookup returns None.
        assert cache_lookup(cache_dir, task_hash) is None


# ---------------------------------------------------------------------------
# cache_stats — non-file glob hit + exception path
# ---------------------------------------------------------------------------


class TestCacheStatsEdgeCases:
    def test_directory_named_result_json_is_skipped(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        # Create a directory matching the */*/result.json glob but which is not a file.
        fake = cache_dir / "ab" / "abcd"
        (fake / "result.json").mkdir(parents=True)
        # — `if not result_path.is_file: continue`.
        stats = cache_stats(cache_dir)
        assert stats["entries"] == 0

    def test_exception_returns_zero_stats(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        def _boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("glob blew up")

        # Force the glob iteration to raise so the except path runs (893-894).
        monkeypatch.setattr(Path, "glob", _boom)
        stats = cache_stats(cache_dir)
        assert stats == {
            "entries": 0,
            "total_size_bytes": 0,
            "oldest": None,
            "newest": None,
        }


# ---------------------------------------------------------------------------
# cache_clear — non-dir shard/entry + exception path
# ---------------------------------------------------------------------------


class TestCacheClearEdgeCases:
    def test_non_directory_shard_is_skipped(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # A loose file at the shard level is not a directory.
        (cache_dir / "stray.txt").write_text("x", encoding="utf-8")
        # — `if not shard_dir.is_dir: continue`.
        assert cache_clear(cache_dir) == 0

    def test_non_directory_entry_is_skipped(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        shard = cache_dir / "ab"
        shard.mkdir(parents=True)
        # A loose file inside a shard directory is not an entry directory.
        (shard / "stray.txt").write_text("x", encoding="utf-8")
        # — `if not entry_dir.is_dir: continue`.
        assert cache_clear(cache_dir) == 0

    def test_real_entry_is_removed_and_counted(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        task_hash = "ab" + "c" * 62
        cache_store(cache_dir, task_hash, _make_result(tmp_path), None)
        assert cache_clear(cache_dir) == 1
        assert cache_lookup(cache_dir, task_hash) is None

    def test_exception_returns_partial_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        def _boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("iterdir blew up")

        # Force iterdir to raise so the except path returns `removed` (919-920).
        monkeypatch.setattr(Path, "iterdir", _boom)
        assert cache_clear(cache_dir) == 0
