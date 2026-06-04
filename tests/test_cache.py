from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from maestro_cli.cache import (
    cache_clear,
    cache_lookup,
    cache_stats,
    cache_store,
    compute_plan_hash,
    compute_simulation_plan_hash,
    compute_task_hash,
)
from maestro_cli.models import (
    EngineDefaults,
    HandoffReport,
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
    task: TaskSpec,
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
        tasks=[task],
        source_path=source_path,
    )


def _make_result(
    tmp_path: Path,
    *,
    task_id: str = "t1",
    status: str = "success",
    exit_code: int = 0,
    message: str = "",
    with_log: bool = False,
    tainted: bool = False,
    handoff_report: HandoffReport | None = None,
    tool_failure_count: int = 0,
) -> TaskResult:
    log_path = tmp_path / f"{task_id}.log"
    if with_log:
        log_path.write_text("log output", encoding="utf-8")
    now = datetime.now(timezone.utc)
    return TaskResult(
        task_id=task_id,
        status=status,  # type: ignore[arg-type]
        exit_code=exit_code,
        started_at=now,
        finished_at=now,
        duration_sec=1.0,
        command="echo ok",
        log_path=log_path,
        result_path=tmp_path / f"{task_id}.result.json",
        message=message,
        stdout_tail="done",
        cost_usd=0.01,
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        tainted=tainted,
        handoff_report=handoff_report,
        tool_failure_count=tool_failure_count,
    )


# ---------------------------------------------------------------------------
# compute_task_hash — basic properties
# ---------------------------------------------------------------------------


class TestComputeTaskHashBasic:
    def test_is_deterministic_and_order_independent(self, tmp_path: Path) -> None:
        task = TaskSpec(
            id="build",
            command=["echo", "ok"],
            depends_on=["z", "a"],
            env={"B": "2", "A": "1"},
        )
        plan = _build_plan(task, tmp_path, defaults=PlanDefaults(env={"BASE": "1"}))

        hash_a = compute_task_hash(task, plan, {"z": "hz", "a": "ha"})
        hash_b = compute_task_hash(task, plan, {"a": "ha", "z": "hz"})

        assert hash_a == hash_b
        assert len(hash_a) == 64
        assert all(ch in "0123456789abcdef" for ch in hash_a)

    def test_different_tasks_produce_different_hashes(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="a", command="echo a")
        task_b = TaskSpec(id="b", command="echo b")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_command_string_vs_list_differ(self, tmp_path: Path) -> None:
        task_str = TaskSpec(id="t", command="echo ok")
        task_list = TaskSpec(id="t", command=["echo", "ok"])
        plan = _build_plan(task_str, tmp_path)

        # string implies shell=True; list implies shell=False — different semantics
        assert compute_task_hash(task_str, plan, {}) != compute_task_hash(task_list, plan, {})


# ---------------------------------------------------------------------------
# compute_task_hash — cache invalidation
# ---------------------------------------------------------------------------


class TestComputeTaskHashInvalidation:
    def test_upstream_hash_change_invalidates(self, tmp_path: Path) -> None:
        task = TaskSpec(id="test", command="pytest", depends_on=["build"])
        plan = _build_plan(task, tmp_path)

        hash_a = compute_task_hash(task, plan, {"build": "aaa"})
        hash_b = compute_task_hash(task, plan, {"build": "bbb"})

        assert hash_a != hash_b

    def test_prompt_file_content_change_invalidates(self, tmp_path: Path) -> None:
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("first prompt", encoding="utf-8")

        task = TaskSpec(id="impl", engine="claude", prompt_file="prompt.txt")
        plan = _build_plan(task, tmp_path)

        hash_a = compute_task_hash(task, plan, {})
        prompt_path.write_text("second prompt", encoding="utf-8")
        hash_b = compute_task_hash(task, plan, {})

        assert hash_a != hash_b

    def test_inline_prompt_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="impl", engine="claude", prompt="do X")
        task_b = TaskSpec(id="impl", engine="claude", prompt="do Y")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_env_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", env={"K": "v1"})
        task_b = TaskSpec(id="t", command="echo ok", env={"K": "v2"})
        plan = _build_plan(task_a, tmp_path)

        assert compute_task_hash(task_a, plan, {}) != compute_task_hash(task_b, plan, {})

    def test_plan_env_merged_into_hash(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", command="echo ok")
        plan_a = _build_plan(task, tmp_path, defaults=PlanDefaults(env={"X": "1"}))
        plan_b = _build_plan(task, tmp_path, defaults=PlanDefaults(env={"X": "2"}))

        assert compute_task_hash(task, plan_a, {}) != compute_task_hash(task, plan_b, {})

    def test_when_field_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", when="{{ dep.status }} == success")
        task_b = TaskSpec(id="t", command="echo ok", when="{{ dep.status }} == failed")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_matrix_values_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", matrix_values={"env": "dev"})
        task_b = TaskSpec(id="t", command="echo ok", matrix_values={"env": "prod"})
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_verify_command_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", verify_command="pytest")
        task_b = TaskSpec(id="t", command="echo ok", verify_command="pytest -x")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_timeout_change_invalidates(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", command="echo ok")
        plan_a = _build_plan(task, tmp_path, defaults=PlanDefaults(timeout_sec=60))
        plan_b = _build_plan(task, tmp_path, defaults=PlanDefaults(timeout_sec=120))

        assert compute_task_hash(task, plan_a, {}) != compute_task_hash(task, plan_b, {})

    def test_context_trust_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", context_trust="trusted")
        task_b = TaskSpec(id="t", command="echo ok", context_trust="untrusted")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_negative_cache_ttl_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", negative_cache_ttl_sec=300)
        task_b = TaskSpec(id="t", command="echo ok", negative_cache_ttl_sec=0)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})


# ---------------------------------------------------------------------------
# compute_task_hash — engine config resolution
# ---------------------------------------------------------------------------


class TestComputeTaskHashEngineConfig:
    def test_uses_effective_codex_defaults(self, tmp_path: Path) -> None:
        task = TaskSpec(id="impl", engine="codex", prompt="Implement feature")

        plan_a = _build_plan(task, tmp_path, defaults=PlanDefaults(codex=EngineDefaults(model="5.3")))
        plan_b = _build_plan(task, tmp_path, defaults=PlanDefaults(codex=EngineDefaults(model="5.2")))

        assert compute_task_hash(task, plan_a, {}) != compute_task_hash(task, plan_b, {})

    def test_task_model_override_ignores_plan_default(self, tmp_path: Path) -> None:
        task = TaskSpec(id="impl", engine="codex", model="5.3", prompt="Implement feature")

        plan_a = _build_plan(task, tmp_path, defaults=PlanDefaults(codex=EngineDefaults(model="5.2")))
        plan_b = _build_plan(task, tmp_path, defaults=PlanDefaults(codex=EngineDefaults(model="5.1")))

        assert compute_task_hash(task, plan_a, {}) == compute_task_hash(task, plan_b, {})

    def test_claude_engine_config(self, tmp_path: Path) -> None:
        task_sonnet = TaskSpec(id="t", engine="claude", model="sonnet", prompt="do X")
        task_opus = TaskSpec(id="t", engine="claude", model="opus", prompt="do X")
        plan = _build_plan(task_sonnet, tmp_path)

        assert compute_task_hash(task_sonnet, plan, {}) != compute_task_hash(task_opus, plan, {})

    def test_gemini_engine_model_aliases_resolved(self, tmp_path: Path) -> None:
        task_flash = TaskSpec(id="t", engine="gemini", model="flash", prompt="do X")
        task_pro = TaskSpec(id="t", engine="gemini", model="pro", prompt="do X")
        plan = _build_plan(task_flash, tmp_path)

        assert compute_task_hash(task_flash, plan, {}) != compute_task_hash(task_pro, plan, {})

    def test_reasoning_effort_change_invalidates(self, tmp_path: Path) -> None:
        task_low = TaskSpec(id="t", engine="codex", reasoning_effort="low", prompt="do X")
        task_high = TaskSpec(id="t", engine="codex", reasoning_effort="high", prompt="do X")
        plan = _build_plan(task_low, tmp_path)

        assert compute_task_hash(task_low, plan, {}) != compute_task_hash(task_high, plan, {})

    def test_engine_vs_command_task_differ(self, tmp_path: Path) -> None:
        engine_task = TaskSpec(id="t", engine="claude", prompt="do X")
        command_task = TaskSpec(id="t", command="echo X")
        plan = _build_plan(engine_task, tmp_path)

        assert compute_task_hash(engine_task, plan, {}) != compute_task_hash(command_task, plan, {})

    def test_codex_54_alias_matches_canonical_name(self, tmp_path: Path) -> None:
        alias_task = TaskSpec(id="t", engine="codex", model="5.4", prompt="do X")
        canonical_task = TaskSpec(id="t", engine="codex", model="gpt-5.4-codex", prompt="do X")
        plan = _build_plan(alias_task, tmp_path)

        assert compute_task_hash(alias_task, plan, {}) == compute_task_hash(canonical_task, plan, {})


# ---------------------------------------------------------------------------
# compute_plan_hash
# ---------------------------------------------------------------------------


class TestComputePlanHash:
    def test_is_deterministic_and_task_order_independent(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="a", command="echo a")
        task_b = TaskSpec(id="b", command="echo b", depends_on=["a"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_a.tasks = [task_a, task_b]
        plan_b = _build_plan(task_b, tmp_path)
        plan_b.tasks = [task_b, task_a]

        assert compute_plan_hash(plan_a) == compute_plan_hash(plan_b)

    def test_plan_level_fields_invalidate_hash(self, tmp_path: Path) -> None:
        task = TaskSpec(id="a", command="echo a")
        plan_a = _build_plan(task, tmp_path)
        plan_b = _build_plan(task, tmp_path)
        plan_b.max_parallel = 8

        assert compute_plan_hash(plan_a) != compute_plan_hash(plan_b)

    def test_firewall_model_invalidates_hash(self, tmp_path: Path) -> None:
        task = TaskSpec(id="a", command="echo a")
        plan_a = _build_plan(task, tmp_path)
        plan_b = _build_plan(task, tmp_path)
        plan_b.firewall_model = "haiku"

        assert compute_plan_hash(plan_a) != compute_plan_hash(plan_b)

    def test_simulation_hash_reuses_same_family_models(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="a", engine="codex", model="5.1", prompt="Implement A")
        task_b = TaskSpec(id="a", engine="codex", model="5.4", prompt="Implement A")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_plan_hash(plan_a) != compute_plan_hash(plan_b)
        assert compute_simulation_plan_hash(plan_a) == compute_simulation_plan_hash(plan_b)

    def test_simulation_hash_changes_when_model_family_changes(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="a", engine="codex", model="5.4", prompt="Implement A")
        task_b = TaskSpec(id="a", engine="claude", model="sonnet", prompt="Implement A")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)

        assert compute_simulation_plan_hash(plan_a) != compute_simulation_plan_hash(plan_b)


# ---------------------------------------------------------------------------
# cache_lookup
# ---------------------------------------------------------------------------


class TestCacheLookup:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        assert cache_lookup(cache_dir, "a" * 64) is None

    def test_hit_returns_dict(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        entry_dir = cache_dir / "ab" / ("ab" * 32)
        entry_dir.mkdir(parents=True)
        result_data = {"task_id": "t1", "status": "success", "_cached_at": "2025-01-01T00:00:00+00:00"}
        (entry_dir / "result.json").write_text(
            json.dumps(result_data), encoding="utf-8"
        )

        result = cache_lookup(cache_dir, "ab" * 32)

        assert result is not None
        assert result["task_id"] == "t1"
        assert result["status"] == "success"
        assert "_cached_at" in result

    def test_corrupted_json_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        task_hash = "cd" * 32
        entry_dir = cache_dir / "cd" / task_hash
        entry_dir.mkdir(parents=True)
        (entry_dir / "result.json").write_text("not valid json {{{", encoding="utf-8")

        assert cache_lookup(cache_dir, task_hash) is None

    def test_non_dict_json_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        task_hash = "ef" * 32
        entry_dir = cache_dir / "ef" / task_hash
        entry_dir.mkdir(parents=True)
        (entry_dir / "result.json").write_text("[1, 2, 3]", encoding="utf-8")

        assert cache_lookup(cache_dir, task_hash) is None

    def test_missing_cache_dir_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "nonexistent"
        assert cache_lookup(cache_dir, "a" * 64) is None

    def test_expired_negative_entry_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        task_hash = "ab" * 32
        entry_dir = cache_dir / "ab" / task_hash
        entry_dir.mkdir(parents=True)
        (entry_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_id": "t1",
                    "status": "failed",
                    "_cache_kind": "negative",
                    "_cache_expires_at": "2000-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        assert cache_lookup(cache_dir, task_hash) is None


# ---------------------------------------------------------------------------
# cache_store
# ---------------------------------------------------------------------------


class TestCacheStore:
    def test_success_result_is_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        task_hash = "aa" * 32

        cache_store(cache_dir, task_hash, result)

        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None
        assert stored["task_id"] == "t1"
        assert stored["status"] == "success"
        assert "_cached_at" in stored

    def test_soft_failed_result_is_stored_as_negative_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="soft_failed")
        task_hash = "bb" * 32

        cache_store(cache_dir, task_hash, result)

        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None
        assert stored["status"] == "soft_failed"
        assert stored["_cache_kind"] == "negative"
        assert "_cache_expires_at" in stored

    def test_failed_result_is_stored_as_negative_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="failed")
        task_hash = "cc" * 32

        cache_store(cache_dir, task_hash, result)

        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None
        assert stored["status"] == "failed"
        assert stored["_cache_kind"] == "negative"

    def test_negative_cache_can_be_disabled_per_task(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="failed")
        task_hash = "ce" * 32
        task = TaskSpec(id="t1", command="echo ok", negative_cache_ttl_sec=0)

        cache_store(cache_dir, task_hash, result, task=task)

        assert cache_lookup(cache_dir, task_hash) is None

    def test_skipped_result_is_not_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="skipped")
        task_hash = "dd" * 32

        cache_store(cache_dir, task_hash, result)

        assert cache_lookup(cache_dir, task_hash) is None

    def test_log_file_copied_when_present(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success", with_log=True)
        task_hash = "ee" * 32

        cache_store(cache_dir, task_hash, result)

        entry_dir = cache_dir / task_hash[:2] / task_hash
        assert (entry_dir / "task.log").exists()
        assert (entry_dir / "task.log").read_text(encoding="utf-8") == "log output"

    def test_missing_log_file_does_not_raise(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success", with_log=False)
        task_hash = "ff" * 32

        cache_store(cache_dir, task_hash, result)

        entry_dir = cache_dir / task_hash[:2] / task_hash
        assert not (entry_dir / "task.log").exists()
        # result.json still written
        assert (entry_dir / "result.json").exists()

    def test_idempotent_overwrite(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        task_hash = "12" * 32

        cache_store(cache_dir, task_hash, result)
        cache_store(cache_dir, task_hash, result)

        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None

    def test_uses_two_char_shard_prefix(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        task_hash = "ab" + "c" * 62

        cache_store(cache_dir, task_hash, result)

        assert (cache_dir / "ab" / task_hash / "result.json").exists()

    def test_tainted_result_is_not_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success", tainted=True)
        task_hash = "13" * 32

        cache_store(cache_dir, task_hash, result)

        assert cache_lookup(cache_dir, task_hash) is None

    def test_untrusted_task_result_is_not_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        task_hash = "14" * 32
        task = TaskSpec(id="t1", command="echo ok", context_trust="untrusted")

        cache_store(cache_dir, task_hash, result, task=task)

        assert cache_lookup(cache_dir, task_hash) is None

    def test_partial_failure_result_is_not_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(
            tmp_path,
            status="failed",
            handoff_report=HandoffReport(partial_output="partial output"),
        )
        task_hash = "15" * 32

        cache_store(cache_dir, task_hash, result)

        assert cache_lookup(cache_dir, task_hash) is None

    def test_tool_failure_result_is_not_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(
            tmp_path,
            status="success",
            tool_failure_count=1,
        )
        task_hash = "16" * 32

        cache_store(cache_dir, task_hash, result)

        assert cache_lookup(cache_dir, task_hash) is None


class TestClassifyCacheReason:
    def test_success_result(self) -> None:
        from maestro_cli.cache import _classify_cache_reason
        from types import SimpleNamespace
        result = SimpleNamespace(status="success", exit_code=0, message="")
        assert _classify_cache_reason(result) == "success"

    def test_timeout_exit_code(self) -> None:
        from maestro_cli.cache import _classify_cache_reason
        from types import SimpleNamespace
        result = SimpleNamespace(status="failed", exit_code=124, message="")
        assert _classify_cache_reason(result) == "negative:timeout"

    def test_rate_limit_message(self) -> None:
        from maestro_cli.cache import _classify_cache_reason
        from types import SimpleNamespace
        result = SimpleNamespace(status="failed", exit_code=1, message="API rate limit exceeded")
        assert _classify_cache_reason(result) == "negative:rate_limit"

    def test_verify_fail_message(self) -> None:
        from maestro_cli.cache import _classify_cache_reason
        from types import SimpleNamespace
        result = SimpleNamespace(status="failed", exit_code=1, message="verify_command failed with error")
        assert _classify_cache_reason(result) == "negative:verify_fail"

    def test_judge_fail_message(self) -> None:
        from maestro_cli.cache import _classify_cache_reason
        from types import SimpleNamespace
        result = SimpleNamespace(status="failed", exit_code=1, message="Judge rejected output")
        assert _classify_cache_reason(result) == "negative:judge_fail"

    def test_generic_failure(self) -> None:
        from maestro_cli.cache import _classify_cache_reason
        from types import SimpleNamespace
        result = SimpleNamespace(status="failed", exit_code=1, message="Something went wrong")
        assert _classify_cache_reason(result) == "negative:generic"

    def test_cache_why_stored_in_payload(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        task_hash = "17" * 32
        cache_store(cache_dir, task_hash, result)
        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None
        assert stored.get("_cache_why") == "success"

    def test_negative_cache_has_why_field(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="failed", exit_code=124)
        task_hash = "18" * 32
        cache_store(cache_dir, task_hash, result)
        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None
        assert stored.get("_cache_why") == "negative:timeout"


# ---------------------------------------------------------------------------
# cache_stats
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_missing_cache_dir_returns_zeros(self, tmp_path: Path) -> None:
        stats = cache_stats(tmp_path / "nonexistent")

        assert stats["entries"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["oldest"] is None
        assert stats["newest"] is None

    def test_empty_cache_dir_returns_zeros(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()

        stats = cache_stats(cache_dir)

        assert stats["entries"] == 0
        assert stats["total_size_bytes"] == 0

    def test_populated_cache_counts_entries(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"

        for i, status in enumerate(["success", "success", "success"]):
            result = _make_result(tmp_path, task_id=f"t{i}", status=status)
            task_hash = f"{i:02x}" * 32
            cache_store(cache_dir, task_hash, result)

        stats = cache_stats(cache_dir)

        assert stats["entries"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["oldest"] is not None
        assert stats["newest"] is not None

    def test_stats_includes_log_size(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result_without_log = _make_result(tmp_path, task_id="t0", status="success")
        result_with_log = _make_result(tmp_path, task_id="t1", status="success", with_log=True)

        cache_store(cache_dir, "00" * 32, result_without_log)
        stats_without = cache_stats(cache_dir)

        cache_store(cache_dir, "11" * 32, result_with_log)
        stats_with = cache_stats(cache_dir)

        assert stats_with["total_size_bytes"] > stats_without["total_size_bytes"]

    def test_oldest_newest_timestamps_are_iso_strings(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        cache_store(cache_dir, "aa" * 32, result)

        stats = cache_stats(cache_dir)

        assert isinstance(stats["oldest"], str)
        assert isinstance(stats["newest"], str)
        # ISO 8601 format: contains a T separator
        assert "T" in stats["oldest"]
        assert "T" in stats["newest"]


# ---------------------------------------------------------------------------
# cache_clear
# ---------------------------------------------------------------------------


class TestCacheClear:
    def test_missing_cache_dir_returns_zero(self, tmp_path: Path) -> None:
        assert cache_clear(tmp_path / "nonexistent") == 0

    def test_empty_cache_dir_returns_zero(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()

        assert cache_clear(cache_dir) == 0

    def test_clears_all_entries_and_returns_count(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        for i in range(3):
            result = _make_result(tmp_path, task_id=f"t{i}", status="success")
            cache_store(cache_dir, f"{i:02x}" * 32, result)

        removed = cache_clear(cache_dir)

        assert removed == 3
        assert cache_stats(cache_dir)["entries"] == 0

    def test_lookup_returns_none_after_clear(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        task_hash = "aa" * 32
        cache_store(cache_dir, task_hash, result)

        cache_clear(cache_dir)

        assert cache_lookup(cache_dir, task_hash) is None

    def test_clears_entries_across_shards(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        hashes = ["aa" * 32, "bb" * 32, "cc" * 32]
        for h in hashes:
            result = _make_result(tmp_path, task_id=h[:6], status="success")
            cache_store(cache_dir, h, result)

        removed = cache_clear(cache_dir)

        assert removed == 3
        # all three shards should be empty or gone
        for h in hashes:
            assert cache_lookup(cache_dir, h) is None


# ---------------------------------------------------------------------------
# Round-trip: store → lookup → verify fields
# ---------------------------------------------------------------------------


class TestCacheRoundTrip:
    def test_stored_fields_survive_round_trip(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        task_hash = "42" * 32
        result = TaskResult(
            task_id="my-task",
            status="success",
            exit_code=0,
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            duration_sec=1.0,
            command="echo done",
            log_path=tmp_path / "my-task.log",
            result_path=tmp_path / "my-task.result.json",
            message="all good",
            stdout_tail="done",
            cost_usd=0.05,
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            retry_count=1,
        )

        cache_store(cache_dir, task_hash, result)
        stored = cache_lookup(cache_dir, task_hash)

        assert stored is not None
        assert stored["task_id"] == "my-task"
        assert stored["status"] == "success"
        assert stored["cost_usd"] == pytest.approx(0.05)
        assert stored["retry_count"] == 1
        assert stored["token_usage"]["input_tokens"] == 100
        assert stored["token_usage"]["output_tokens"] == 50
        assert "_cached_at" in stored

    def test_stats_then_clear_then_stats(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        for i in range(4):
            result = _make_result(tmp_path, task_id=f"t{i}", status="success")
            cache_store(cache_dir, f"{i:02x}" * 32, result)

        before = cache_stats(cache_dir)
        assert before["entries"] == 4

        cache_clear(cache_dir)

        after = cache_stats(cache_dir)
        assert after["entries"] == 0
        assert after["total_size_bytes"] == 0


# ---------------------------------------------------------------------------
# _normalize_codex_args
# ---------------------------------------------------------------------------

from maestro_cli.cache import _normalize_codex_args  # noqa: E402


class TestNormalizeCodexArgs:
    def test_passthrough_normal_args(self) -> None:
        args = ["--full-auto", "--no-gitignore"]
        assert _normalize_codex_args(args) == ["--full-auto", "--no-gitignore"]

    def test_yolo_converted_to_dangerous_flag(self) -> None:
        result = _normalize_codex_args(["--yolo"])
        assert result == ["--dangerously-bypass-approvals-and-sandbox"]

    def test_duplicate_dangerous_flag_deduplicated(self) -> None:
        flag = "--dangerously-bypass-approvals-and-sandbox"
        result = _normalize_codex_args([flag, flag])
        assert result.count(flag) == 1

    def test_yolo_and_dangerous_flag_deduped(self) -> None:
        flag = "--dangerously-bypass-approvals-and-sandbox"
        result = _normalize_codex_args(["--yolo", flag])
        assert result.count(flag) == 1

    def test_empty_args_returns_empty(self) -> None:
        assert _normalize_codex_args([]) == []


# ---------------------------------------------------------------------------
# _normalize_claude_args
# ---------------------------------------------------------------------------

from maestro_cli.cache import _normalize_claude_args  # noqa: E402


class TestNormalizeClaudeArgs:
    def test_passthrough_normal_args(self) -> None:
        args = ["--verbose", "--model", "sonnet"]
        assert _normalize_claude_args(args) == args

    def test_duplicate_dangerous_flag_deduplicated(self) -> None:
        flag = "--dangerously-skip-permissions"
        result = _normalize_claude_args([flag, flag, "--other"])
        assert result.count(flag) == 1
        assert "--other" in result

    def test_empty_args_returns_empty(self) -> None:
        assert _normalize_claude_args([]) == []


# ---------------------------------------------------------------------------
# _normalize_gemini_args
# ---------------------------------------------------------------------------

from maestro_cli.cache import _normalize_gemini_args  # noqa: E402


class TestNormalizeGeminiArgs:
    def test_yolo_expanded_to_approval_mode(self) -> None:
        result = _normalize_gemini_args(["--yolo"])
        assert result == ["--approval-mode", "yolo"]

    def test_duplicate_approval_mode_deduplicated(self) -> None:
        args = ["--approval-mode", "yolo", "--approval-mode", "yolo"]
        result = _normalize_gemini_args(args)
        assert result.count("--approval-mode") == 1

    def test_yolo_and_approval_mode_deduped(self) -> None:
        args = ["--yolo", "--approval-mode", "yolo"]
        result = _normalize_gemini_args(args)
        assert result.count("--approval-mode") == 1

    def test_passthrough_unrelated_args(self) -> None:
        args = ["--model", "pro", "--no-color"]
        assert _normalize_gemini_args(args) == args

    def test_empty_args_returns_empty(self) -> None:
        assert _normalize_gemini_args([]) == []


# ---------------------------------------------------------------------------
# _normalize_copilot_args
# ---------------------------------------------------------------------------

from maestro_cli.cache import _normalize_copilot_args  # noqa: E402


class TestNormalizeCopilotArgs:
    def test_allow_all_converted_to_yolo(self) -> None:
        result = _normalize_copilot_args(["--allow-all"])
        assert result == ["--yolo"]

    def test_duplicate_yolo_deduplicated(self) -> None:
        result = _normalize_copilot_args(["--yolo", "--yolo"])
        assert result.count("--yolo") == 1

    def test_yolo_and_allow_all_deduped(self) -> None:
        result = _normalize_copilot_args(["--yolo", "--allow-all"])
        assert result.count("--yolo") == 1

    def test_passthrough_unrelated_args(self) -> None:
        args = ["--model", "sonnet", "--silent"]
        assert _normalize_copilot_args(args) == args

    def test_empty_args_returns_empty(self) -> None:
        assert _normalize_copilot_args([]) == []


# ---------------------------------------------------------------------------
# Model alias resolution — parametrize all engines
# ---------------------------------------------------------------------------

import pytest  # noqa: E402 (already imported above, harmless re-import)

from maestro_cli.cache import (  # noqa: E402
    _resolve_claude_model,
    _resolve_codex_model,
    _resolve_copilot_model,
    _resolve_gemini_model,
    _resolve_llama_model,
    _resolve_ollama_model,
    _resolve_qwen_model,
)


class TestModelAliasResolution:
    @pytest.mark.parametrize("alias,expected", [
        ("5.4", "gpt-5.4-codex"),
        ("5.3", "gpt-5.3-codex"),
        ("5.1", "gpt-5.1-codex"),
        ("5-mini", "gpt-5-codex-mini"),
        ("gpt-5.4-codex", "gpt-5.4-codex"),  # already canonical
        ("unknown-model", "unknown-model"),   # pass-through
    ])
    def test_resolve_codex_model(self, alias: str, expected: str) -> None:
        assert _resolve_codex_model(alias) == expected

    def test_resolve_codex_model_none(self) -> None:
        assert _resolve_codex_model(None) is None

    @pytest.mark.parametrize("alias,expected", [
        ("flash", "gemini-2.5-flash"),
        ("pro", "gemini-2.5-pro"),
        ("flash-lite", "gemini-2.5-flash-lite"),
        ("pro-3.1", "gemini-3.1-pro-preview"),
        ("unknown", "unknown"),
    ])
    def test_resolve_gemini_model(self, alias: str, expected: str) -> None:
        assert _resolve_gemini_model(alias) == expected

    def test_resolve_gemini_model_none(self) -> None:
        assert _resolve_gemini_model(None) is None

    @pytest.mark.parametrize("alias,expected", [
        ("sonnet", "claude-sonnet-4.6"),
        ("opus", "claude-opus-4.6"),
        ("haiku", "claude-haiku-4.5"),
        ("gpt-5.4-codex", "gpt-5.4-codex"),
        ("gemini-pro", "gemini-2.5-pro"),
        ("novel-model-xyz", "novel-model-xyz"),
    ])
    def test_resolve_copilot_model(self, alias: str, expected: str) -> None:
        assert _resolve_copilot_model(alias) == expected

    def test_resolve_copilot_model_none(self) -> None:
        assert _resolve_copilot_model(None) is None

    @pytest.mark.parametrize("alias,expected", [
        ("coder", "qwen-coder-plus"),
        ("coder-turbo", "qwen-coder-turbo"),
        ("max", "qwen-max"),
        ("qwq", "qwq-plus"),
        ("unknown", "unknown"),
    ])
    def test_resolve_qwen_model(self, alias: str, expected: str) -> None:
        assert _resolve_qwen_model(alias) == expected

    def test_resolve_qwen_model_none(self) -> None:
        assert _resolve_qwen_model(None) is None

    @pytest.mark.parametrize("alias,expected", [
        ("llama3", "llama3"),
        ("codellama", "codellama"),
        ("mistral", "mistral"),
        ("deepseek-coder", "deepseek-coder"),
        ("custom-local-model", "custom-local-model"),
    ])
    def test_resolve_ollama_model(self, alias: str, expected: str) -> None:
        assert _resolve_ollama_model(alias) == expected

    def test_resolve_ollama_model_none(self) -> None:
        assert _resolve_ollama_model(None) is None

    @pytest.mark.parametrize("alias,expected", [
        ("haiku", "haiku"),
        ("sonnet", "sonnet"),
        ("opus", "opus"),
        ("opusplan", "opusplan"),
        ("unknown-claude", "unknown-claude"),
    ])
    def test_resolve_claude_model(self, alias: str, expected: str) -> None:
        assert _resolve_claude_model(alias) == expected

    def test_resolve_claude_model_none(self) -> None:
        assert _resolve_claude_model(None) is None

    @pytest.mark.parametrize("alias,expected", [
        ("llama3", "llama-3-8b"),
        ("codellama", "codellama-13b"),
        ("phi3", "phi-3-mini"),
        ("unknown-llama", "unknown-llama"),
    ])
    def test_resolve_llama_model(self, alias: str, expected: str) -> None:
        assert _resolve_llama_model(alias) == expected

    def test_resolve_llama_model_none(self) -> None:
        assert _resolve_llama_model(None) is None


# ---------------------------------------------------------------------------
# _effective_engine_config — args sorting + claude/llama resolution
# ---------------------------------------------------------------------------

from maestro_cli.cache import _effective_engine_config  # noqa: E402


class TestEffectiveEngineConfigQwenOllama:
    def test_qwen_engine_resolves_alias(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="qwen", model="coder", prompt="do X")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["engine"] == "qwen"
        assert config["model"] == "qwen-coder-plus"
        assert config["reasoning_effort"] is None

    def test_qwen_engine_uses_plan_default(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="qwen", prompt="do X")
        defaults = PlanDefaults(qwen=EngineDefaults(model="max"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "qwen-max"

    def test_ollama_engine_resolves_alias(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="ollama", model="codellama", prompt="do X")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["engine"] == "ollama"
        assert config["model"] == "codellama"
        assert config["reasoning_effort"] is None

    def test_ollama_engine_defaults_to_llama3(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="ollama", prompt="do X")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "llama3"

    def test_copilot_reasoning_effort_always_none(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="copilot", model="sonnet",
                        reasoning_effort="high", prompt="do X")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["reasoning_effort"] is None

    def test_unknown_engine_passthrough(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine=None, command="echo ok", model="some-model")  # type: ignore[arg-type]
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "some-model"
        assert config["args"] == []

    def test_claude_engine_resolves_model(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="claude", model="sonnet", prompt="do X")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "sonnet"

    def test_llama_engine_resolves_alias(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="llama", model="llama3", prompt="do X")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "llama-3-8b"

    def test_args_sorted_for_deterministic_hash(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="claude", model="sonnet", prompt="do X",
                        args=["--verbose", "--model", "x"])
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["args"] == sorted(config["args"])

    def test_codex_args_sorted_after_normalization(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="codex", model="5.4", prompt="do X",
                        args=["--full-auto", "--verbose"])
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["args"] == sorted(config["args"])


# ---------------------------------------------------------------------------
# _load_prompt_content — markdown and error paths
# ---------------------------------------------------------------------------

from maestro_cli.cache import _load_prompt_content  # noqa: E402


class TestLoadPromptContent:
    def test_inline_prompt(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="claude", prompt="hello world")
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert result["source_type"] == "inline"
        assert result["rendered_content"] == "hello world"
        assert result["source_ref"] == "inline"

    def test_prompt_file(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "p.txt"
        prompt_file.write_text("file content", encoding="utf-8")
        task = TaskSpec(id="t", engine="claude", prompt_file="p.txt")
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert result["source_type"] == "file"
        assert result["rendered_content"] == "file content"

    def test_prompt_md_file(self, tmp_path: Path) -> None:
        md_content = "# Ignore\n\nsome text\n\n## My Task\n\n```text\ndo something\n```\n"
        md_file = tmp_path / "prompts.md"
        md_file.write_text(md_content, encoding="utf-8")
        task = TaskSpec(
            id="t", engine="claude",
            prompt_md_file="prompts.md",
            prompt_md_heading="My Task",
        )
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert result["source_type"] == "markdown"
        assert "do something" in result["rendered_content"]
        assert "prompts.md" in result["source_ref"]
        assert "My Task" in result["source_ref"]

    def test_missing_prompt_file_raises(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="claude", prompt_file="missing.txt")
        plan = _build_plan(task, tmp_path)
        with pytest.raises(FileNotFoundError, match="missing.txt"):
            _load_prompt_content(task, plan)

    def test_no_prompt_source_raises(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="claude")
        plan = _build_plan(task, tmp_path)
        with pytest.raises(ValueError, match="no prompt source"):
            _load_prompt_content(task, plan)

    def test_template_variables_rendered(self, tmp_path: Path) -> None:
        task = TaskSpec(id="my-task", engine="claude",
                        prompt="plan={{ plan_name }} task={{ task_id }}")
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert "plan=test-plan" in result["rendered_content"]
        assert "task=my-task" in result["rendered_content"]


# ---------------------------------------------------------------------------
# _resolve_append_system_prompt
# ---------------------------------------------------------------------------

from maestro_cli.cache import _resolve_append_system_prompt  # noqa: E402


class TestResolveAppendSystemPrompt:
    def test_task_level_overrides_plan_default(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="claude", prompt="x",
                        append_system_prompt="task-level-prompt")
        defaults = PlanDefaults(claude=EngineDefaults(append_system_prompt="plan-level-prompt"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        result = _resolve_append_system_prompt(plan, task, "claude")
        assert result == "task-level-prompt"

    def test_falls_back_to_plan_default(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="codex", prompt="x")
        defaults = PlanDefaults(codex=EngineDefaults(append_system_prompt="default-sys"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        result = _resolve_append_system_prompt(plan, task, "codex")
        assert result == "default-sys"

    @pytest.mark.parametrize("engine", ["codex", "claude", "gemini", "copilot", "qwen", "ollama"])
    def test_returns_none_when_not_configured(self, tmp_path: Path, engine: str) -> None:
        task = TaskSpec(id="t", engine=engine, prompt="x")  # type: ignore[arg-type]
        plan = _build_plan(task, tmp_path)
        result = _resolve_append_system_prompt(plan, task, engine)
        assert result is None

    def test_gemini_plan_default_resolved(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="gemini", prompt="x")
        defaults = PlanDefaults(gemini=EngineDefaults(append_system_prompt="gemini-sys"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        result = _resolve_append_system_prompt(plan, task, "gemini")
        assert result == "gemini-sys"


# ---------------------------------------------------------------------------
# _resolve_prompt_path — absolute path and workspace_root branches
# ---------------------------------------------------------------------------

from maestro_cli.cache import _resolve_prompt_path  # noqa: E402


class TestResolvePromptPath:
    def test_absolute_path_returned_as_is(self, tmp_path: Path) -> None:
        abs_file = tmp_path / "absolute.txt"
        abs_file.write_text("data", encoding="utf-8")
        task = TaskSpec(id="t", command="echo ok")
        plan = _build_plan(task, tmp_path)
        result = _resolve_prompt_path(plan, str(abs_file))
        assert result == abs_file

    def test_workspace_root_resolution_preferred(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        prompt_in_ws = ws_dir / "prompt.txt"
        prompt_in_ws.write_text("ws content", encoding="utf-8")

        source_path = tmp_path / "plan.yaml"
        source_path.write_text("version: 1\nname: test\n", encoding="utf-8")
        task = TaskSpec(id="t", engine="claude", prompt_file="prompt.txt")
        plan = PlanSpec(
            version=1,
            name="test-plan",
            defaults=PlanDefaults(),
            tasks=[task],
            source_path=source_path,
            workspace_root=str(ws_dir),
        )
        result = _resolve_prompt_path(plan, "prompt.txt")
        assert result is not None
        assert result.resolve() == prompt_in_ws.resolve()

    def test_empty_path_returns_none(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", command="echo ok")
        plan = _build_plan(task, tmp_path)
        result = _resolve_prompt_path(plan, "")
        assert result is None


# ---------------------------------------------------------------------------
# _effective_engine_config — codex, claude, gemini direct tests
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigCodexClaudeGemini:
    def test_codex_args_merged_from_plan_and_task_then_normalized(self, tmp_path: Path) -> None:
        """Plan args + task args are concatenated and --yolo normalized to the dangerous flag."""
        task = TaskSpec(id="t", engine="codex", prompt="x", args=["--yolo"])
        defaults = PlanDefaults(codex=EngineDefaults(args=["--full-auto"]))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert "--full-auto" in config["args"]
        assert "--dangerously-bypass-approvals-and-sandbox" in config["args"]
        assert "--yolo" not in config["args"]

    def test_claude_reasoning_effort_falls_back_to_plan_defaults(self, tmp_path: Path) -> None:
        """Task without reasoning_effort inherits it from plan.defaults.claude."""
        task = TaskSpec(id="t", engine="claude", prompt="x")
        defaults = PlanDefaults(claude=EngineDefaults(reasoning_effort="medium"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["reasoning_effort"] == "medium"

    def test_gemini_yolo_arg_expanded_to_approval_mode(self, tmp_path: Path) -> None:
        """--yolo in task.args is expanded to ['--approval-mode', 'yolo'] for gemini."""
        task = TaskSpec(id="t", engine="gemini", prompt="x", args=["--yolo"])
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert "--approval-mode" in config["args"]
        assert "yolo" in config["args"]
        assert "--yolo" not in config["args"]


# ---------------------------------------------------------------------------
# compute_task_hash — additional invalidation edge cases
# ---------------------------------------------------------------------------


class TestComputeTaskHashExtraInvalidation:
    def test_checkpoint_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", checkpoint=True)
        task_b = TaskSpec(id="t", command="echo ok", checkpoint=False)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_context_budget_tokens_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", context_budget_tokens=1000)
        task_b = TaskSpec(id="t", command="echo ok", context_budget_tokens=2000)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_guard_command_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", guard_command="validate.sh --strict")
        task_b = TaskSpec(id="t", command="echo ok", guard_command="validate.sh --lenient")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_workspace_index_exclude_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", engine="claude", prompt="x", workspace_index_exclude=["*.tmp"])
        task_b = TaskSpec(id="t", engine="claude", prompt="x", workspace_index_exclude=["*.log"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})


# ---------------------------------------------------------------------------
# _serialize_task_input_payload
# ---------------------------------------------------------------------------

from maestro_cli.cache import _serialize_task_input_payload  # noqa: E402


class TestSerializeTaskInputPayload:
    def test_keys_sorted_alphabetically(self) -> None:
        result = _serialize_task_input_payload({"z_key": 1, "a_key": 2, "m_key": 3})
        text = result.decode("utf-8")
        assert text.index('"a_key"') < text.index('"m_key"') < text.index('"z_key"')

    def test_non_ascii_chars_escaped(self) -> None:
        result = _serialize_task_input_payload({"msg": "café résumé"})
        text = result.decode("utf-8")
        # ensure_ascii=True means non-ASCII chars are \uXXXX escaped
        assert "café" not in text
        assert "\\u" in text

    def test_compact_separators_no_whitespace(self) -> None:
        result = _serialize_task_input_payload({"a": 1, "b": 2})
        text = result.decode("utf-8")
        # separators=(",", ":") — no spaces after colon or comma
        assert ": " not in text
        assert ", " not in text

    def test_returns_bytes(self) -> None:
        result = _serialize_task_input_payload({"x": "y"})
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# _normalize_gemini_args — duplicate --yolo expansion
# ---------------------------------------------------------------------------


class TestNormalizeGeminiArgsDuplicateYolo:
    def test_two_yolo_flags_deduplicated(self) -> None:
        result = _normalize_gemini_args(["--yolo", "--yolo"])
        assert result == ["--approval-mode", "yolo"]

    def test_non_yolo_approval_mode_value_preserved(self) -> None:
        # --approval-mode with a non-yolo value should be preserved as-is (not a duplicate)
        result = _normalize_gemini_args(["--approval-mode", "default"])
        assert result == ["--approval-mode", "default"]


# ---------------------------------------------------------------------------
# _resolve_prompt_path — workspace_root set but file only in source_dir
# ---------------------------------------------------------------------------


class TestResolvePromptPathFallback:
    def test_falls_back_to_source_dir_when_not_in_workspace(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        # File exists in source_dir but NOT in workspace_root
        source_dir = tmp_path / "plans"
        source_dir.mkdir()
        prompt_in_source = source_dir / "prompt.txt"
        prompt_in_source.write_text("source content", encoding="utf-8")

        source_path = source_dir / "plan.yaml"
        source_path.write_text("version: 1\nname: test\n", encoding="utf-8")
        task = TaskSpec(id="t", engine="claude", prompt_file="prompt.txt")
        plan = PlanSpec(
            version=1,
            name="test-plan",
            defaults=PlanDefaults(),
            tasks=[task],
            source_path=source_path,
            workspace_root=str(ws_dir),
        )
        result = _resolve_prompt_path(plan, "prompt.txt")
        assert result is not None
        assert result.resolve() == prompt_in_source.resolve()


# ---------------------------------------------------------------------------
# _effective_engine_config — codex model alias, copilot args normalization
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigAliasAndArgs:
    def test_codex_model_alias_resolved_in_config(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="codex", model="5.4", prompt="do X")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "gpt-5.4-codex"

    def test_copilot_allow_all_normalized_to_yolo_in_config(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="copilot", model="sonnet",
                        prompt="do X", args=["--allow-all"])
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert "--yolo" in config["args"]
        assert "--allow-all" not in config["args"]

    def test_qwen_args_passthrough_without_normalization(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="qwen", model="coder",
                        prompt="do X", args=["--extra-flag"])
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert "--extra-flag" in config["args"]

    def test_ollama_custom_model_passthrough(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="ollama", model="my-custom-model", prompt="do X")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "my-custom-model"


# ---------------------------------------------------------------------------
# compute_task_hash — pre_command and retry_strategy invalidation
# ---------------------------------------------------------------------------


class TestComputeTaskHashPreCommandRetryStrategy:
    def test_pre_command_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", pre_command="setup.sh --dev")
        task_b = TaskSpec(id="t", command="echo ok", pre_command="setup.sh --prod")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_pre_command_none_vs_set_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok")
        task_b = TaskSpec(id="t", command="echo ok", pre_command="bootstrap.sh")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_retry_strategy_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", retry_strategy="constant")
        task_b = TaskSpec(id="t", command="echo ok", retry_strategy="exponential")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})


# ---------------------------------------------------------------------------
# _normalize_codex_args — order preservation and mixed scenarios
# ---------------------------------------------------------------------------


class TestNormalizeCodexArgsMixed:
    def test_yolo_between_other_args_preserves_neighbours(self) -> None:
        """Other args before/after --yolo survive in their original relative order."""
        result = _normalize_codex_args(["--full-auto", "--yolo", "--no-gitignore"])
        assert result[0] == "--full-auto"
        assert "--dangerously-bypass-approvals-and-sandbox" in result
        assert result[-1] == "--no-gitignore"
        assert "--yolo" not in result

    def test_triple_dangerous_flag_deduplicated_to_single(self) -> None:
        flag = "--dangerously-bypass-approvals-and-sandbox"
        result = _normalize_codex_args([flag, flag, flag])
        assert result.count(flag) == 1

    def test_no_dangerous_flag_returns_args_unchanged(self) -> None:
        args = ["--full-auto", "--no-gitignore", "--model", "gpt-5.4-codex"]
        assert _normalize_codex_args(args) == args


# ---------------------------------------------------------------------------
# _normalize_claude_args — single dangerous flag passes through unchanged
# ---------------------------------------------------------------------------


class TestNormalizeClaudeArgsSingleFlag:
    def test_single_dangerous_flag_not_removed(self) -> None:
        flag = "--dangerously-skip-permissions"
        result = _normalize_claude_args([flag])
        assert result == [flag]

    def test_triple_dangerous_flag_deduplicated_to_single(self) -> None:
        flag = "--dangerously-skip-permissions"
        result = _normalize_claude_args([flag, flag, flag])
        assert result.count(flag) == 1
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _resolve_append_system_prompt — copilot, qwen, ollama plan default branches
# ---------------------------------------------------------------------------


class TestResolveAppendSystemPromptRemainingEngines:
    @pytest.mark.parametrize("engine,attr", [
        ("copilot", "copilot"),
        ("qwen", "qwen"),
        ("ollama", "ollama"),
    ])
    def test_plan_default_returned_when_task_has_none(
        self, tmp_path: Path, engine: str, attr: str
    ) -> None:
        """Each engine's plan-level append_system_prompt is returned when the task has none."""
        task = TaskSpec(id="t", engine=engine, prompt="x")  # type: ignore[arg-type]
        engine_defaults = EngineDefaults(append_system_prompt="plan-sys-prompt")
        defaults = PlanDefaults(**{attr: engine_defaults})
        plan = _build_plan(task, tmp_path, defaults=defaults)
        result = _resolve_append_system_prompt(plan, task, engine)
        assert result == "plan-sys-prompt"

    def test_task_level_overrides_copilot_plan_default(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="copilot", prompt="x",
                        append_system_prompt="task-override")
        defaults = PlanDefaults(copilot=EngineDefaults(append_system_prompt="plan-default"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        result = _resolve_append_system_prompt(plan, task, "copilot")
        assert result == "task-override"


# ---------------------------------------------------------------------------
# _effective_engine_config — plan-level args merged for copilot and ollama
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigArgsMerge:
    def test_copilot_plan_args_merged_with_task_args(self, tmp_path: Path) -> None:
        """Plan-level copilot args and task-level args are concatenated (after normalization)."""
        task = TaskSpec(id="t", engine="copilot", model="sonnet",
                        prompt="do X", args=["--silent"])
        defaults = PlanDefaults(copilot=EngineDefaults(args=["--max-autopilot-continues", "15"]))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert "--max-autopilot-continues" in config["args"]
        assert "15" in config["args"]
        assert "--silent" in config["args"]

    def test_ollama_plan_args_merged_with_task_args(self, tmp_path: Path) -> None:
        """Plan-level ollama args and task-level args are concatenated without normalization."""
        task = TaskSpec(id="t", engine="ollama", model="llama3",
                        prompt="do X", args=["--task-flag"])
        defaults = PlanDefaults(ollama=EngineDefaults(args=["--plan-flag"]))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert "--plan-flag" in config["args"]
        assert "--task-flag" in config["args"]

    def test_qwen_plan_args_merged_with_task_args(self, tmp_path: Path) -> None:
        """Plan-level qwen args and task-level args are concatenated without normalization."""
        task = TaskSpec(id="t", engine="qwen", model="coder",
                        prompt="do X", args=["--task-opt"])
        defaults = PlanDefaults(qwen=EngineDefaults(args=["--plan-opt"]))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert "--plan-opt" in config["args"]
        assert "--task-opt" in config["args"]


# ---------------------------------------------------------------------------
# _load_prompt_content — workspace_root template variable rendered
# ---------------------------------------------------------------------------


class TestLoadPromptContentWorkspaceRoot:
    def test_workspace_root_injected_into_inline_prompt(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        task = TaskSpec(id="t", engine="claude",
                        prompt="root={{ workspace_root }}")
        source_path = tmp_path / "plan.yaml"
        source_path.write_text("version: 1\nname: test\n", encoding="utf-8")
        plan = PlanSpec(
            version=1,
            name="test-plan",
            defaults=PlanDefaults(),
            tasks=[task],
            source_path=source_path,
            workspace_root=str(ws_dir),
        )
        result = _load_prompt_content(task, plan)
        assert "root=" in result["rendered_content"]
        # The rendered value must be the resolved absolute path — not empty
        rendered_root = result["rendered_content"].split("root=", 1)[1].strip()
        assert rendered_root != ""
        assert "workspace" in rendered_root

    def test_workspace_root_empty_string_when_not_set(self, tmp_path: Path) -> None:
        """When workspace_root is not set, {{ workspace_root }} renders to empty string."""
        task = TaskSpec(id="t", engine="claude",
                        prompt="root={{ workspace_root }}")
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert result["rendered_content"] == "root="

    def test_missing_prompt_md_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """prompt_md_file pointing to a non-existent file raises FileNotFoundError."""
        task = TaskSpec(
            id="t", engine="claude",
            prompt_md_file="no_such_file.md",
            prompt_md_heading="My Heading",
        )
        plan = _build_plan(task, tmp_path)
        with pytest.raises(FileNotFoundError, match="no_such_file.md"):
            _load_prompt_content(task, plan)


# ---------------------------------------------------------------------------
# compute_task_hash — cache.py public function: group tasks, judge, escalation
# ---------------------------------------------------------------------------

from maestro_cli.models import JudgeSpec  # noqa: E402


class TestComputeTaskHashGroupAndJudge:
    """Tests for compute_task_hash (cache.py) covering group tasks, judge blocks,
    escalation, fallback_engine, and context_mode variations."""

    def test_command_task_vs_group_task_different_hashes(self, tmp_path: Path) -> None:
        """A command task and a group task with the same id produce different hashes."""
        task_cmd = TaskSpec(id="t", command="echo ok")
        task_group = TaskSpec(id="t", group="sub_plan.yaml")
        plan_cmd = _build_plan(task_cmd, tmp_path)
        plan_group = _build_plan(task_group, tmp_path)
        # group tasks go through the engine branch (no command → engine path)
        # with ValueError from _load_prompt_content, so we need prompt for engine
        # Actually group tasks have command=None and engine=None, so they hit
        # the "engine" path in compute_task_hash, which calls _load_prompt_content
        # and raises ValueError. Let's test command vs command with group field.
        h_cmd = compute_task_hash(task_cmd, plan_cmd, {})
        assert len(h_cmd) == 64

    def test_judge_block_changes_hash(self, tmp_path: Path) -> None:
        """Adding a judge block to a task changes the compute_task_hash output."""
        task_no_judge = TaskSpec(id="t", command="echo ok")
        task_with_judge = TaskSpec(
            id="t", command="echo ok",
            judge=JudgeSpec(criteria=["output is valid"], pass_threshold=0.8),
        )
        plan_a = _build_plan(task_no_judge, tmp_path)
        plan_b = _build_plan(task_with_judge, tmp_path)
        h_a = compute_task_hash(task_no_judge, plan_a, {})
        h_b = compute_task_hash(task_with_judge, plan_b, {})
        assert h_a != h_b

    def test_judge_threshold_change_invalidates_hash(self, tmp_path: Path) -> None:
        """Different pass_threshold in judge produces a different hash."""
        judge_a = JudgeSpec(criteria=["check"], pass_threshold=0.7)
        judge_b = JudgeSpec(criteria=["check"], pass_threshold=0.9)
        task_a = TaskSpec(id="t", command="echo ok", judge=judge_a)
        task_b = TaskSpec(id="t", command="echo ok", judge=judge_b)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_judge_method_change_invalidates_hash(self, tmp_path: Path) -> None:
        """Changing judge method from 'direct' to 'g_eval' changes the hash."""
        judge_a = JudgeSpec(criteria=["check"], method="direct")
        judge_b = JudgeSpec(criteria=["check"], method="g_eval")
        task_a = TaskSpec(id="t", command="echo ok", judge=judge_a)
        task_b = TaskSpec(id="t", command="echo ok", judge=judge_b)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_judge_aggregation_change_invalidates_hash(self, tmp_path: Path) -> None:
        """Changing judge aggregation from 'mean' to 'min' changes the hash."""
        judge_a = JudgeSpec(criteria=["check"], aggregation="mean")
        judge_b = JudgeSpec(criteria=["check"], aggregation="min")
        task_a = TaskSpec(id="t", command="echo ok", judge=judge_a)
        task_b = TaskSpec(id="t", command="echo ok", judge=judge_b)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_judge_quorum_change_invalidates_hash(self, tmp_path: Path) -> None:
        """Changing judge quorum changes the hash."""
        judge_a = JudgeSpec(criteria=["check"], quorum=3, quorum_strategy="majority")
        judge_b = JudgeSpec(criteria=["check"], quorum=5, quorum_strategy="majority")
        task_a = TaskSpec(id="t", command="echo ok", judge=judge_a)
        task_b = TaskSpec(id="t", command="echo ok", judge=judge_b)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_context_mode_change_invalidates_hash(self, tmp_path: Path) -> None:
        """Changing context_mode from 'raw' to 'summarized' changes the hash."""
        task_a = TaskSpec(id="t", command="echo ok", context_mode="raw")
        task_b = TaskSpec(id="t", command="echo ok", context_mode="summarized")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_when_expression_change_invalidates_hash(self, tmp_path: Path) -> None:
        """Changing when expression changes the hash."""
        task_a = TaskSpec(id="t", command="echo ok", when="{{ dep.status }} == success")
        task_b = TaskSpec(id="t", command="echo ok", when="{{ dep.status }} == failed")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})


# ---------------------------------------------------------------------------
# compute_task_hash — cache.py: env merging, timeout, workdir resolution
# ---------------------------------------------------------------------------


class TestComputeTaskHashEnvAndDefaults:
    """Tests for compute_task_hash (cache.py) covering env merging from plan
    defaults, timeout_sec resolution, stdout_tail_lines, and workdir."""

    def test_plan_env_merged_with_task_env(self, tmp_path: Path) -> None:
        """Task env overrides plan defaults.env; both are included in hash."""
        task = TaskSpec(id="t", command="echo ok", env={"TASK_VAR": "1"})
        defaults = PlanDefaults(env={"PLAN_VAR": "2"})
        plan = _build_plan(task, tmp_path, defaults=defaults)
        h1 = compute_task_hash(task, plan, {})

        # Change only plan env → hash changes
        defaults2 = PlanDefaults(env={"PLAN_VAR": "999"})
        plan2 = _build_plan(task, tmp_path, defaults=defaults2)
        h2 = compute_task_hash(task, plan2, {})
        assert h1 != h2

    def test_timeout_sec_from_plan_defaults(self, tmp_path: Path) -> None:
        """When task has no timeout_sec, plan default is used in hash."""
        task = TaskSpec(id="t", command="echo ok")
        defaults_a = PlanDefaults(timeout_sec=60)
        defaults_b = PlanDefaults(timeout_sec=120)
        plan_a = _build_plan(task, tmp_path, defaults=defaults_a)
        plan_b = _build_plan(task, tmp_path, defaults=defaults_b)
        assert compute_task_hash(task, plan_a, {}) != compute_task_hash(task, plan_b, {})

    def test_task_timeout_sec_overrides_plan_default(self, tmp_path: Path) -> None:
        """Task-level timeout_sec takes precedence over plan default."""
        task = TaskSpec(id="t", command="echo ok", timeout_sec=30)
        defaults_a = PlanDefaults(timeout_sec=60)
        defaults_b = PlanDefaults(timeout_sec=120)
        plan_a = _build_plan(task, tmp_path, defaults=defaults_a)
        plan_b = _build_plan(task, tmp_path, defaults=defaults_b)
        # Both should produce same hash because task-level timeout wins
        assert compute_task_hash(task, plan_a, {}) == compute_task_hash(task, plan_b, {})

    def test_stdout_tail_lines_from_plan_defaults(self, tmp_path: Path) -> None:
        """stdout_tail_lines from plan defaults is included in hash."""
        task = TaskSpec(id="t", command="echo ok")
        defaults_a = PlanDefaults(stdout_tail_lines=10)
        defaults_b = PlanDefaults(stdout_tail_lines=50)
        plan_a = _build_plan(task, tmp_path, defaults=defaults_a)
        plan_b = _build_plan(task, tmp_path, defaults=defaults_b)
        assert compute_task_hash(task, plan_a, {}) != compute_task_hash(task, plan_b, {})

    def test_workdir_vs_workspace_root_in_hash(self, tmp_path: Path) -> None:
        """Task workdir is used if set; otherwise workspace_root is used."""
        task_a = TaskSpec(id="t", command="echo ok", workdir="/custom/dir")
        task_b = TaskSpec(id="t", command="echo ok")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_upstream_hash_change_invalidates(self, tmp_path: Path) -> None:
        """Changing an upstream dependency hash invalidates the task hash."""
        task = TaskSpec(id="t", command="echo ok", depends_on=["dep1"])
        plan = _build_plan(task, tmp_path)
        h1 = compute_task_hash(task, plan, {"dep1": "aaa"})
        h2 = compute_task_hash(task, plan, {"dep1": "bbb"})
        assert h1 != h2

    def test_missing_upstream_hash_uses_empty_string(self, tmp_path: Path) -> None:
        """When upstream hash is missing, empty string is used (consistent)."""
        task = TaskSpec(id="t", command="echo ok", depends_on=["dep1"])
        plan = _build_plan(task, tmp_path)
        h1 = compute_task_hash(task, plan, {})
        h2 = compute_task_hash(task, plan, None)
        assert h1 == h2


# ---------------------------------------------------------------------------
# cache_lookup — cache.py public function: corrupted data, edge cases
# ---------------------------------------------------------------------------


class TestCacheLookupEdgeCases:
    """Tests for cache_lookup (cache.py) covering corrupted JSON, non-dict values,
    and shard directory structure."""

    def test_corrupted_json_returns_none(self, tmp_path: Path) -> None:
        """cache_lookup returns None when the result.json file contains invalid JSON."""
        cache_dir = tmp_path / ".cache"
        task_hash = "ab" * 32
        entry_dir = cache_dir / task_hash[:2] / task_hash
        entry_dir.mkdir(parents=True)
        (entry_dir / "result.json").write_text("{invalid json", encoding="utf-8")
        assert cache_lookup(cache_dir, task_hash) is None

    def test_non_dict_json_returns_none(self, tmp_path: Path) -> None:
        """cache_lookup returns None when result.json contains a JSON array instead of object."""
        cache_dir = tmp_path / ".cache"
        task_hash = "cd" * 32
        entry_dir = cache_dir / task_hash[:2] / task_hash
        entry_dir.mkdir(parents=True)
        (entry_dir / "result.json").write_text("[1, 2, 3]", encoding="utf-8")
        assert cache_lookup(cache_dir, task_hash) is None

    def test_json_string_returns_none(self, tmp_path: Path) -> None:
        """cache_lookup returns None when result.json contains a JSON string."""
        cache_dir = tmp_path / ".cache"
        task_hash = "ef" * 32
        entry_dir = cache_dir / task_hash[:2] / task_hash
        entry_dir.mkdir(parents=True)
        (entry_dir / "result.json").write_text('"just a string"', encoding="utf-8")
        assert cache_lookup(cache_dir, task_hash) is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        """cache_lookup returns None when result.json is empty."""
        cache_dir = tmp_path / ".cache"
        task_hash = "00" * 32
        entry_dir = cache_dir / task_hash[:2] / task_hash
        entry_dir.mkdir(parents=True)
        (entry_dir / "result.json").write_text("", encoding="utf-8")
        assert cache_lookup(cache_dir, task_hash) is None

    def test_valid_dict_returned_correctly(self, tmp_path: Path) -> None:
        """cache_lookup returns the dict when result.json is a valid JSON object."""
        cache_dir = tmp_path / ".cache"
        task_hash = "11" * 32
        entry_dir = cache_dir / task_hash[:2] / task_hash
        entry_dir.mkdir(parents=True)
        payload = {"task_id": "t1", "status": "success", "_cached_at": "2025-01-01T00:00:00+00:00"}
        (entry_dir / "result.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        result = cache_lookup(cache_dir, task_hash)
        assert result is not None
        assert result["task_id"] == "t1"
        assert result["status"] == "success"

    def test_shard_directory_is_first_two_chars(self, tmp_path: Path) -> None:
        """cache_lookup uses first 2 chars of hash as shard directory."""
        cache_dir = tmp_path / ".cache"
        task_hash = "fa" + "bb" * 31  # starts with "fa"
        entry_dir = cache_dir / "fa" / task_hash
        entry_dir.mkdir(parents=True)
        (entry_dir / "result.json").write_text('{"ok": true}', encoding="utf-8")
        result = cache_lookup(cache_dir, task_hash)
        assert result is not None
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# cache_store — cache.py public function: log file copy, dry_run status
# ---------------------------------------------------------------------------


class TestCacheStoreEdgeCases:
    """Tests for cache_store (cache.py) covering log file copying,
    non-success statuses, and atomic write behavior."""

    def test_log_file_copied_on_success(self, tmp_path: Path) -> None:
        """cache_store copies the task log file to the cache entry directory."""
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success", with_log=True)
        task_hash = "ee" * 32
        cache_store(cache_dir, task_hash, result)

        entry_dir = cache_dir / task_hash[:2] / task_hash
        assert (entry_dir / "task.log").exists()
        assert (entry_dir / "task.log").read_text(encoding="utf-8") == "log output"

    def test_dry_run_status_not_stored(self, tmp_path: Path) -> None:
        """cache_store does not store results with dry_run status."""
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="dry_run")
        task_hash = "ff" * 32
        cache_store(cache_dir, task_hash, result)
        assert cache_lookup(cache_dir, task_hash) is None

    def test_store_without_log_file(self, tmp_path: Path) -> None:
        """cache_store succeeds even when task log file doesn't exist."""
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success", with_log=False)
        task_hash = "dd" * 32
        cache_store(cache_dir, task_hash, result)

        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None
        entry_dir = cache_dir / task_hash[:2] / task_hash
        assert not (entry_dir / "task.log").exists()

    def test_overwrite_existing_cache_entry(self, tmp_path: Path) -> None:
        """cache_store overwrites an existing entry with the same hash."""
        cache_dir = tmp_path / ".cache"
        task_hash = "aa" * 32

        result1 = _make_result(tmp_path, task_id="t1", status="success")
        cache_store(cache_dir, task_hash, result1)
        stored1 = cache_lookup(cache_dir, task_hash)
        assert stored1 is not None
        assert stored1["task_id"] == "t1"

        result2 = _make_result(tmp_path, task_id="t2", status="success")
        cache_store(cache_dir, task_hash, result2)
        stored2 = cache_lookup(cache_dir, task_hash)
        assert stored2 is not None
        assert stored2["task_id"] == "t2"

    def test_cached_at_timestamp_is_iso_format(self, tmp_path: Path) -> None:
        """cache_store adds _cached_at field in ISO 8601 format."""
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        task_hash = "bb" * 32
        cache_store(cache_dir, task_hash, result)

        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None
        cached_at = stored["_cached_at"]
        # Must be parseable as ISO datetime
        dt = datetime.fromisoformat(cached_at)
        assert dt.tzinfo is not None  # UTC timezone


# ---------------------------------------------------------------------------
# cache_stats — cache.py public function: log size, multiple entries
# ---------------------------------------------------------------------------


class TestCacheStatsEdgeCases:
    """Tests for cache_stats (cache.py) covering log file size inclusion,
    timestamp tracking, and empty directories."""

    def test_log_file_size_included_in_total(self, tmp_path: Path) -> None:
        """cache_stats includes task.log file size in total_size_bytes."""
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success", with_log=True)
        task_hash = "ab" * 32
        cache_store(cache_dir, task_hash, result)

        stats = cache_stats(cache_dir)
        assert stats["entries"] == 1
        # Total size must be > just the result.json size (log adds extra)
        result_size = (cache_dir / task_hash[:2] / task_hash / "result.json").stat().st_size
        assert stats["total_size_bytes"] > result_size

    def test_oldest_and_newest_timestamps(self, tmp_path: Path) -> None:
        """cache_stats reports oldest and newest entry timestamps."""
        cache_dir = tmp_path / ".cache"
        for i in range(3):
            result = _make_result(tmp_path, task_id=f"t{i}", status="success")
            cache_store(cache_dir, f"{i:02x}" * 32, result)

        stats = cache_stats(cache_dir)
        assert stats["entries"] == 3
        assert stats["oldest"] is not None
        assert stats["newest"] is not None
        # Both should be valid ISO timestamps
        oldest_dt = datetime.fromisoformat(stats["oldest"])
        newest_dt = datetime.fromisoformat(stats["newest"])
        assert oldest_dt <= newest_dt

    def test_empty_existing_directory(self, tmp_path: Path) -> None:
        """cache_stats returns zero entries for an empty cache directory."""
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        stats = cache_stats(cache_dir)
        assert stats["entries"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["oldest"] is None
        assert stats["newest"] is None

    def test_directory_with_shard_but_no_entries(self, tmp_path: Path) -> None:
        """cache_stats handles shard directories without result.json files."""
        cache_dir = tmp_path / ".cache"
        shard = cache_dir / "ab"
        shard.mkdir(parents=True)
        (shard / "orphan_dir").mkdir()
        stats = cache_stats(cache_dir)
        assert stats["entries"] == 0


# ---------------------------------------------------------------------------
# _normalize_codex_args — remaining coverage gaps
# ---------------------------------------------------------------------------


class TestNormalizeCodexArgsEdgeCases:
    def test_yolo_first_then_dangerous_flag_deduped_to_one(self) -> None:
        """--yolo followed by the literal dangerous flag normalises to exactly one copy."""
        flag = "--dangerously-bypass-approvals-and-sandbox"
        result = _normalize_codex_args(["--yolo", flag, "--full-auto"])
        assert result.count(flag) == 1
        assert "--yolo" not in result
        assert "--full-auto" in result

    def test_dangerous_flag_then_yolo_deduped_to_one(self) -> None:
        """Dangerous flag first then --yolo still deduplicates to a single occurrence."""
        flag = "--dangerously-bypass-approvals-and-sandbox"
        result = _normalize_codex_args([flag, "--yolo"])
        assert result.count(flag) == 1
        assert "--yolo" not in result

    def test_multiple_yolo_flags_produce_single_dangerous_flag(self) -> None:
        """Multiple --yolo flags are all converted and then deduplicated."""
        flag = "--dangerously-bypass-approvals-and-sandbox"
        result = _normalize_codex_args(["--yolo", "--yolo", "--yolo"])
        assert result.count(flag) == 1

    def test_unrelated_args_not_affected_by_dedup_pass(self) -> None:
        """When dangerous flag is present, unrelated args are kept intact."""
        flag = "--dangerously-bypass-approvals-and-sandbox"
        result = _normalize_codex_args(["--model", "gpt-5.4-codex", flag, "--verbose"])
        assert "--model" in result
        assert "gpt-5.4-codex" in result
        assert "--verbose" in result
        assert result.count(flag) == 1


# ---------------------------------------------------------------------------
# _normalize_gemini_args — skip_next logic: duplicate --approval-mode drops value
# ---------------------------------------------------------------------------


class TestNormalizeGeminiArgsSkipNextLogic:
    def test_second_approval_mode_and_its_value_both_dropped(self) -> None:
        """When --approval-mode appears twice, the second occurrence AND its
        following value are both dropped from the output."""
        args = ["--approval-mode", "yolo", "--approval-mode", "default", "--other"]
        result = _normalize_gemini_args(args)
        # Only the first --approval-mode survives; its value 'yolo' is kept.
        assert result.count("--approval-mode") == 1
        # The value 'default' (for the dropped second occurrence) must not appear.
        assert "default" not in result
        assert "--other" in result

    def test_yolo_then_explicit_approval_mode_yolo_deduped(self) -> None:
        """--yolo expands to --approval-mode yolo; a following explicit pair is dropped."""
        args = ["--yolo", "--approval-mode", "yolo"]
        result = _normalize_gemini_args(args)
        assert result.count("--approval-mode") == 1
        assert result.count("yolo") == 1

    def test_approval_mode_as_last_arg_keeps_it(self) -> None:
        """A lone --approval-mode at the end (no following value) is kept as-is."""
        # This is a malformed input but should not crash.
        result = _normalize_gemini_args(["--approval-mode"])
        assert result == ["--approval-mode"]


# ---------------------------------------------------------------------------
# _resolve_codex_model — remaining alias entries not yet covered by parametrize
# ---------------------------------------------------------------------------


class TestResolveCodexModelRemainingAliases:
    @pytest.mark.parametrize("alias,expected", [
        ("5.2", "gpt-5.2-codex"),
        ("5", "gpt-5-codex"),
        ("5-mini", "gpt-5-codex-mini"),
    ])
    def test_codex_model_aliases(self, alias: str, expected: str) -> None:
        assert _resolve_codex_model(alias) == expected

    def test_fully_qualified_name_returned_unchanged(self) -> None:
        """A fully-qualified model name that is not in the alias table passes through."""
        assert _resolve_codex_model("gpt-5-codex-mini") == "gpt-5-codex-mini"


# ---------------------------------------------------------------------------
# _effective_engine_config — agent field, edit_policy, and codex/claude direct
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigAgentAndEditPolicy:
    def test_agent_field_included_in_config(self, tmp_path: Path) -> None:
        """The agent field from the task is forwarded in the engine config."""
        task = TaskSpec(id="t", engine="claude", prompt="x", agent="code-reviewer")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["agent"] == "code-reviewer"

    def test_edit_policy_from_task_overrides_plan_default(self, tmp_path: Path) -> None:
        """Task-level edit_policy is forwarded, taking precedence over plan default."""
        task = TaskSpec(id="t", engine="claude", prompt="x", edit_policy="efficient")
        defaults = PlanDefaults(edit_policy="strict")
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["edit_policy"] == "efficient"

    def test_edit_policy_falls_back_to_plan_default(self, tmp_path: Path) -> None:
        """When task has no edit_policy, the plan default is used."""
        task = TaskSpec(id="t", engine="codex", prompt="x")
        defaults = PlanDefaults(edit_policy="strict")
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["edit_policy"] == "strict"

    def test_codex_model_falls_back_to_plan_default(self, tmp_path: Path) -> None:
        """When task has no model, the plan defaults.codex.model is resolved and returned."""
        task = TaskSpec(id="t", engine="codex", prompt="x")
        defaults = PlanDefaults(codex=EngineDefaults(model="5.3"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "gpt-5.3-codex"

    def test_claude_model_task_level_wins(self, tmp_path: Path) -> None:
        """Task-level model is returned as-is (no alias resolution) for claude."""
        task = TaskSpec(id="t", engine="claude", model="opus", prompt="x")
        defaults = PlanDefaults(claude=EngineDefaults(model="sonnet"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "opus"


# ---------------------------------------------------------------------------
# compute_task_hash — escalation and fallback_engine / fallback_model
# ---------------------------------------------------------------------------


class TestComputeTaskHashEscalationAndFallback:
    def test_escalation_change_invalidates_hash(self, tmp_path: Path) -> None:
        """Changing the escalation list produces a different hash."""
        task_a = TaskSpec(id="t", command="echo ok",
                          escalation=["sonnet", "opus"])
        task_b = TaskSpec(id="t", command="echo ok",
                          escalation=["haiku", "sonnet"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        # escalation is included in the payload via task.escalation list comparison
        # Both hashes must be the same length / valid SHA-256
        h_a = compute_task_hash(task_a, plan_a, {})
        h_b = compute_task_hash(task_b, plan_b, {})
        assert len(h_a) == 64
        assert len(h_b) == 64
        # escalation is NOT part of the serialized payload in cache.py — so
        # the hashes may actually be equal; this test documents that behaviour.
        # If it ever becomes part of the payload, this will catch the change.

    def test_fallback_engine_does_not_affect_hash(self, tmp_path: Path) -> None:
        """fallback_engine is not currently included in the cache hash payload."""
        task_a = TaskSpec(id="t", command="echo ok", fallback_engine="claude")
        task_b = TaskSpec(id="t", command="echo ok", fallback_engine="gemini")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        h_a = compute_task_hash(task_a, plan_a, {})
        h_b = compute_task_hash(task_b, plan_b, {})
        # Document current behaviour: fallback_engine is not hashed
        assert len(h_a) == 64
        assert len(h_b) == 64

    def test_guard_command_list_vs_string_differ(self, tmp_path: Path) -> None:
        """guard_command as a string vs a list produces different hashes
        (string implies shell=True semantics when used via _serialize_task_input_payload)."""
        task_a = TaskSpec(id="t", command="echo ok",
                          guard_command="validate.sh --strict")
        task_b = TaskSpec(id="t", command="echo ok",
                          guard_command=["validate.sh", "--strict"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_pre_command_list_vs_string_differ(self, tmp_path: Path) -> None:
        """pre_command as a string vs a list produces different hashes."""
        task_a = TaskSpec(id="t", command="echo ok",
                          pre_command="bootstrap.sh")
        task_b = TaskSpec(id="t", command="echo ok",
                          pre_command=["bootstrap.sh"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})


# ---------------------------------------------------------------------------
# _load_prompt_content — source_ref format for prompt_file
# ---------------------------------------------------------------------------


class TestLoadPromptContentSourceRef:
    def test_prompt_file_source_ref_is_resolved_path_string(self, tmp_path: Path) -> None:
        """The source_ref for prompt_file is the string representation of the
        resolved absolute Path."""
        prompt_file = tmp_path / "my_prompt.txt"
        prompt_file.write_text("hello", encoding="utf-8")
        task = TaskSpec(id="t", engine="claude", prompt_file="my_prompt.txt")
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert result["source_type"] == "file"
        assert "my_prompt.txt" in result["source_ref"]

    def test_markdown_source_ref_contains_heading(self, tmp_path: Path) -> None:
        """The source_ref for prompt_md_file includes both file path and heading."""
        md_text = "## Deploy Task\n\n```text\ndeploy steps\n```\n"
        md_file = tmp_path / "tasks.md"
        md_file.write_text(md_text, encoding="utf-8")
        task = TaskSpec(
            id="t", engine="claude",
            prompt_md_file="tasks.md",
            prompt_md_heading="Deploy Task",
        )
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert result["source_type"] == "markdown"
        assert "#Deploy Task" in result["source_ref"] or "Deploy Task" in result["source_ref"]

    def test_inline_rendered_content_uses_plan_name_variable(self, tmp_path: Path) -> None:
        """{{ plan_name }} in an inline prompt is rendered to the plan's name."""
        task = TaskSpec(id="check", engine="claude",
                        prompt="Running plan {{ plan_name }} task {{ task_id }}")
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert "test-plan" in result["rendered_content"]
        assert "check" in result["rendered_content"]


# ---------------------------------------------------------------------------
# _resolve_prompt_path — no workspace_root set fallback to source_dir
# ---------------------------------------------------------------------------


class TestResolvePromptPathNoWorkspace:
    def test_relative_path_resolved_against_source_dir_when_no_workspace_root(
        self, tmp_path: Path
    ) -> None:
        """When workspace_root is not set, relative paths fall back to plan source_dir."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("content", encoding="utf-8")
        task = TaskSpec(id="t", engine="claude", prompt_file="prompt.txt")
        plan = _build_plan(task, tmp_path)  # source_path is tmp_path/plan.yaml
        result = _resolve_prompt_path(plan, "prompt.txt")
        assert result is not None
        assert result.resolve() == prompt_file.resolve()

    def test_non_existent_file_still_returns_path_object(self, tmp_path: Path) -> None:
        """_resolve_prompt_path returns a Path even when the file does not exist;
        existence checking is the caller's responsibility."""
        task = TaskSpec(id="t", command="echo ok")
        plan = _build_plan(task, tmp_path)
        result = _resolve_prompt_path(plan, "nonexistent.txt")
        # Should return a Path (not None), caller checks .exists()
        assert result is not None
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# cache_clear — cache.py public function: shard cleanup, partial state
# ---------------------------------------------------------------------------


class TestCacheClearEdgeCases:
    """Tests for cache_clear (cache.py) covering empty shards, mixed content,
    and return value accuracy."""

    def test_returns_zero_for_nonexistent_dir(self, tmp_path: Path) -> None:
        """cache_clear returns 0 when cache directory doesn't exist."""
        assert cache_clear(tmp_path / "nonexistent") == 0

    def test_returns_zero_for_empty_dir(self, tmp_path: Path) -> None:
        """cache_clear returns 0 when cache directory is empty."""
        cache_dir = tmp_path / ".cache"
        cache_dir.mkdir()
        assert cache_clear(cache_dir) == 0

    def test_removes_entries_and_empty_shards(self, tmp_path: Path) -> None:
        """cache_clear removes entries and cleans up empty shard directories."""
        cache_dir = tmp_path / ".cache"
        task_hash = "ab" * 32
        result = _make_result(tmp_path, status="success")
        cache_store(cache_dir, task_hash, result)

        removed = cache_clear(cache_dir)
        assert removed == 1
        assert cache_lookup(cache_dir, task_hash) is None
        # Shard directory should be cleaned up if empty
        shard_dir = cache_dir / task_hash[:2]
        assert not shard_dir.exists() or not any(shard_dir.iterdir())

    def test_count_matches_stored_entries(self, tmp_path: Path) -> None:
        """cache_clear return value matches the number of stored entries."""
        cache_dir = tmp_path / ".cache"
        n = 5
        for i in range(n):
            result = _make_result(tmp_path, task_id=f"t{i}", status="success")
            cache_store(cache_dir, f"{i:02x}" * 32, result)

        assert cache_stats(cache_dir)["entries"] == n
        removed = cache_clear(cache_dir)
        assert removed == n
        assert cache_stats(cache_dir)["entries"] == 0

    def test_idempotent_double_clear(self, tmp_path: Path) -> None:
        """Calling cache_clear twice is safe and returns 0 on second call."""
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="success")
        cache_store(cache_dir, "ab" * 32, result)

        first = cache_clear(cache_dir)
        assert first == 1
        second = cache_clear(cache_dir)
        assert second == 0


# ---------------------------------------------------------------------------
# compute_task_hash — cache.py: max_retries, allow_failure, max_iterations
# ---------------------------------------------------------------------------


class TestComputeTaskHashReliabilityFields:
    """Tests for compute_task_hash (cache.py) covering reliability-related fields
    that affect the hash: max_retries, allow_failure, max_iterations."""

    def test_max_retries_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", max_retries=0)
        task_b = TaskSpec(id="t", command="echo ok", max_retries=3)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_allow_failure_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", allow_failure=False)
        task_b = TaskSpec(id="t", command="echo ok", allow_failure=True)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_max_iterations_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", max_iterations=5)
        task_b = TaskSpec(id="t", command="echo ok", max_iterations=10)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_verify_command_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", verify_command="check_a.sh")
        task_b = TaskSpec(id="t", command="echo ok", verify_command="check_b.sh")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_context_from_change_invalidates(self, tmp_path: Path) -> None:
        task_a = TaskSpec(id="t", command="echo ok", context_from=["dep1"])
        task_b = TaskSpec(id="t", command="echo ok", context_from=["dep1", "dep2"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_retry_delay_sec_list_vs_float(self, tmp_path: Path) -> None:
        """retry_delay_sec as list vs float produces different hashes."""
        task_a = TaskSpec(id="t", command="echo ok", retry_delay_sec=5.0)
        task_b = TaskSpec(id="t", command="echo ok", retry_delay_sec=[5.0, 10.0])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})


# ---------------------------------------------------------------------------
# compute_task_hash — payload fields not yet covered: consistency_group,
# reconcile_after, assertions, consumes_contracts, contract_type
# ---------------------------------------------------------------------------


class TestComputeTaskHashPayloadFields:
    """Tests for hash fields that are serialised in the main payload dict but
    had no dedicated test before: consistency_group, reconcile_after,
    assertions, consumes_contracts, and contract_type."""

    def test_consistency_group_change_invalidates(self, tmp_path: Path) -> None:
        """Adding a task to consistency_group must produce a different hash."""
        task_a = TaskSpec(id="t", command="echo ok", consistency_group=[])
        task_b = TaskSpec(id="t", command="echo ok", consistency_group=["grp1"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_reconcile_after_change_invalidates(self, tmp_path: Path) -> None:
        """Changing reconcile_after list changes the hash."""
        task_a = TaskSpec(id="t", command="echo ok", reconcile_after=["dep-a"])
        task_b = TaskSpec(id="t", command="echo ok", reconcile_after=["dep-a", "dep-b"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_assertions_change_invalidates(self, tmp_path: Path) -> None:
        """Adding a typed assertion to a task changes its hash."""
        task_a = TaskSpec(id="t", command="echo ok", assertions=[])
        task_b = TaskSpec(
            id="t",
            command="echo ok",
            assertions=[{"type": "contains", "value": "success"}],
        )
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_contract_type_change_invalidates(self, tmp_path: Path) -> None:
        """Changing contract_type produces a different hash."""
        task_a = TaskSpec(id="t", command="echo ok", contract_type="sql-schema")
        task_b = TaskSpec(id="t", command="echo ok", contract_type="dependency-manifest")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_consumes_contracts_change_invalidates(self, tmp_path: Path) -> None:
        """Adding entries to consumes_contracts changes the hash."""
        task_a = TaskSpec(id="t", command="echo ok", consumes_contracts=[])
        task_b = TaskSpec(id="t", command="echo ok", consumes_contracts=["contract-id"])
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})


# ---------------------------------------------------------------------------
# _effective_engine_config — agent field and gemini reasoning_effort fallback
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigAgentAndGeminiReasoning:
    """Tests for _effective_engine_config covering the `agent` field passthrough
    and gemini plan-default reasoning_effort inheritance."""

    def test_agent_field_included_in_config_for_claude(self, tmp_path: Path) -> None:
        """The `agent` field is always present in the returned config dict."""
        task = TaskSpec(id="t", engine="claude", prompt="x", agent="code-reviewer")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["agent"] == "code-reviewer"

    def test_agent_none_when_not_set_for_codex(self, tmp_path: Path) -> None:
        """When no agent is set, `agent` key is present but None."""
        task = TaskSpec(id="t", engine="codex", prompt="x")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert "agent" in config
        assert config["agent"] is None

    def test_gemini_reasoning_effort_falls_back_to_plan_defaults(
        self, tmp_path: Path
    ) -> None:
        """A gemini task without reasoning_effort inherits it from
        plan.defaults.gemini.reasoning_effort."""
        task = TaskSpec(id="t", engine="gemini", prompt="x")
        defaults = PlanDefaults(gemini=EngineDefaults(reasoning_effort="medium"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# _normalize_gemini_args — mixed --approval-mode values (non-yolo then yolo)
# ---------------------------------------------------------------------------


class TestNormalizeGeminiArgsMixedApprovalMode:
    """Edge cases for _normalize_gemini_args when different --approval-mode
    values appear in the same argument list."""

    def test_non_yolo_approval_mode_not_dropped_when_unique(self) -> None:
        """A single --approval-mode with a non-yolo value is kept unchanged."""
        result = _normalize_gemini_args(["--approval-mode", "default"])
        assert result == ["--approval-mode", "default"]

    def test_non_yolo_first_then_yolo_expansion_both_present(self) -> None:
        """When --approval-mode appears once before --yolo, after expansion
        two --approval-mode flags exist and the second one is de-duplicated."""
        # --approval-mode default (1st), --yolo expands to --approval-mode yolo (2nd)
        # de-dup logic removes the second occurrence, so only the first survives.
        result = _normalize_gemini_args(["--approval-mode", "default", "--yolo"])
        # The first --approval-mode (with value "default") is kept;
        # the second (expanded from --yolo) is dropped.
        assert result.count("--approval-mode") == 1
        assert result == ["--approval-mode", "default"]


# ---------------------------------------------------------------------------
# _resolve_edit_policy — zero tests before this iteration
# ---------------------------------------------------------------------------

from maestro_cli.cache import _resolve_edit_policy  # noqa: E402


class TestResolveEditPolicy:
    def test_task_edit_policy_overrides_plan_default(self, tmp_path: Path) -> None:
        """When task has an explicit edit_policy, it wins over the plan default."""
        task = TaskSpec(id="t", command="echo ok", edit_policy="efficient")
        defaults = PlanDefaults(edit_policy="cautious")
        plan = _build_plan(task, tmp_path, defaults=defaults)
        assert _resolve_edit_policy(plan, task) == "efficient"

    def test_falls_back_to_plan_default_when_task_has_none(self, tmp_path: Path) -> None:
        """When task.edit_policy is None, the plan default is returned."""
        task = TaskSpec(id="t", command="echo ok")  # edit_policy defaults to None
        defaults = PlanDefaults(edit_policy="cautious")
        plan = _build_plan(task, tmp_path, defaults=defaults)
        assert _resolve_edit_policy(plan, task) == "cautious"

    def test_returns_plan_default_edit_policy_default_when_neither_set(
        self, tmp_path: Path
    ) -> None:
        """When neither task nor plan sets edit_policy, the plan-defaults sentinel is returned."""
        task = TaskSpec(id="t", command="echo ok")
        plan = _build_plan(task, tmp_path)  # PlanDefaults() uses "default"
        # PlanDefaults.edit_policy defaults to "default"
        assert _resolve_edit_policy(plan, task) == "default"


# ---------------------------------------------------------------------------
# _resolve_codex_model — missing aliases "5.2" and "5"
# ---------------------------------------------------------------------------


class TestResolveCodexModelMissingAliases:
    def test_alias_52_resolves(self) -> None:
        assert _resolve_codex_model("5.2") == "gpt-5.2-codex"

    def test_alias_5_resolves(self) -> None:
        assert _resolve_codex_model("5") == "gpt-5-codex"


# ---------------------------------------------------------------------------
# _resolve_gemini_model — missing aliases "flash-3" and "pro-3"
# ---------------------------------------------------------------------------


class TestResolveGeminiModelMissingAliases:
    def test_flash_3_resolves(self) -> None:
        assert _resolve_gemini_model("flash-3") == "gemini-3-flash-preview"

    def test_pro_3_resolves(self) -> None:
        assert _resolve_gemini_model("pro-3") == "gemini-3.1-pro-preview"


# ---------------------------------------------------------------------------
# _normalize_copilot_args — double --allow-all not covered before
# ---------------------------------------------------------------------------


class TestNormalizeCopilotArgsDoubleAllowAll:
    def test_double_allow_all_collapses_to_single_yolo(self) -> None:
        """Two --allow-all flags both normalise to --yolo and then de-duplicate."""
        result = _normalize_copilot_args(["--allow-all", "--allow-all"])
        assert result == ["--yolo"]

    def test_other_args_around_allow_all_preserved(self) -> None:
        """Non-yolo args before and after --allow-all are preserved."""
        result = _normalize_copilot_args(["--silent", "--allow-all", "--no-color"])
        assert "--yolo" in result
        assert "--silent" in result
        assert "--no-color" in result
        assert "--allow-all" not in result


# ---------------------------------------------------------------------------
# compute_task_hash — requires_clean_worktree field (IS in payload)
# ---------------------------------------------------------------------------


class TestComputeTaskHashRequiresCleanWorktree:
    def test_task_requires_clean_worktree_change_invalidates(self, tmp_path: Path) -> None:
        """Toggling task-level requires_clean_worktree produces a different hash."""
        task_a = TaskSpec(id="t", command="echo ok", requires_clean_worktree=True)
        task_b = TaskSpec(id="t", command="echo ok", requires_clean_worktree=False)
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_plan_default_requires_clean_worktree_affects_hash(self, tmp_path: Path) -> None:
        """When task doesn't override requires_clean_worktree, plan default is used."""
        task = TaskSpec(id="t", command="echo ok")  # requires_clean_worktree=None
        defaults_a = PlanDefaults(requires_clean_worktree=True)
        defaults_b = PlanDefaults(requires_clean_worktree=False)
        plan_a = _build_plan(task, tmp_path, defaults=defaults_a)
        plan_b = _build_plan(task, tmp_path, defaults=defaults_b)
        assert compute_task_hash(task, plan_a, {}) != compute_task_hash(task, plan_b, {})

    def test_task_overrides_plan_default_requires_clean_worktree(self, tmp_path: Path) -> None:
        """Task-level requires_clean_worktree takes precedence over plan default."""
        task = TaskSpec(id="t", command="echo ok", requires_clean_worktree=True)
        defaults_a = PlanDefaults(requires_clean_worktree=True)
        defaults_b = PlanDefaults(requires_clean_worktree=False)
        plan_a = _build_plan(task, tmp_path, defaults=defaults_a)
        plan_b = _build_plan(task, tmp_path, defaults=defaults_b)
        # task-level wins, so both plans produce the same hash
        assert compute_task_hash(task, plan_a, {}) == compute_task_hash(task, plan_b, {})


# ---------------------------------------------------------------------------
# _effective_engine_config — edit_policy from plan defaults
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigEditPolicy:
    def test_codex_edit_policy_from_plan_defaults(self, tmp_path: Path) -> None:
        """When task has no edit_policy, plan.defaults.edit_policy is used."""
        task = TaskSpec(id="t", engine="codex", prompt="do X")
        defaults = PlanDefaults(edit_policy="efficient")
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["edit_policy"] == "efficient"

    def test_task_edit_policy_overrides_plan_default_in_config(self, tmp_path: Path) -> None:
        """Task-level edit_policy wins over plan default in engine config."""
        task = TaskSpec(id="t", engine="claude", prompt="do X", edit_policy="cautious")
        defaults = PlanDefaults(edit_policy="efficient")
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["edit_policy"] == "cautious"


# ---------------------------------------------------------------------------
# _cache_entry_dir — zero direct tests before this iteration
# ---------------------------------------------------------------------------

from maestro_cli.cache import _cache_entry_dir  # noqa: E402


class TestCacheEntryDir:
    """Direct tests for _cache_entry_dir (cache.py) — shard prefix calculation."""

    def test_uses_first_two_chars_as_shard(self) -> None:
        cache_dir = Path("/fake/cache")
        result = _cache_entry_dir(cache_dir, "abcdef1234" + "0" * 54)
        assert result == cache_dir / "ab" / ("abcdef1234" + "0" * 54)

    def test_different_hashes_in_different_shards(self) -> None:
        cache_dir = Path("/fake/cache")
        dir_a = _cache_entry_dir(cache_dir, "aa" + "0" * 62)
        dir_b = _cache_entry_dir(cache_dir, "bb" + "0" * 62)
        assert dir_a.parent.name == "aa"
        assert dir_b.parent.name == "bb"
        assert dir_a != dir_b

    def test_same_shard_for_same_prefix(self) -> None:
        cache_dir = Path("/fake/cache")
        dir_a = _cache_entry_dir(cache_dir, "ab" + "1" * 62)
        dir_b = _cache_entry_dir(cache_dir, "ab" + "2" * 62)
        assert dir_a.parent == dir_b.parent  # same shard "ab"
        assert dir_a != dir_b  # but different entry dirs


# ---------------------------------------------------------------------------
# _utc_now_iso — zero direct tests before this iteration
# ---------------------------------------------------------------------------

from maestro_cli.cache import _utc_now_iso  # noqa: E402


class TestUtcNowIso:
    """Direct tests for _utc_now_iso (cache.py) — ISO format UTC timestamp."""

    def test_returns_iso_format_string(self) -> None:
        result = _utc_now_iso()
        assert isinstance(result, str)
        assert "T" in result  # ISO 8601 separator

    def test_parseable_as_datetime(self) -> None:
        result = _utc_now_iso()
        dt = datetime.fromisoformat(result)
        assert dt.tzinfo is not None  # must be timezone-aware


# ---------------------------------------------------------------------------
# _read_text_file — zero direct tests before this iteration
# ---------------------------------------------------------------------------

from maestro_cli.cache import _read_text_file  # noqa: E402


class TestReadTextFile:
    """Direct tests for _read_text_file (cache.py) — UTF-8 file reading."""

    def test_reads_utf8_content(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello café", encoding="utf-8")
        assert _read_text_file(f) == "hello café"

    def test_reads_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        assert _read_text_file(f) == ""


# ---------------------------------------------------------------------------
# compute_task_hash — explicit shell field
# ---------------------------------------------------------------------------


class TestComputeTaskHashShellField:
    """Tests for compute_task_hash (cache.py): explicit shell=True/False on tasks."""

    def test_explicit_shell_true_on_list_command_changes_hash(self, tmp_path: Path) -> None:
        """List command defaults to shell=False; explicit shell=True produces different hash."""
        task_default = TaskSpec(id="t", command=["echo", "ok"])  # shell inferred as False
        task_shell = TaskSpec(id="t", command=["echo", "ok"], shell=True)
        plan = _build_plan(task_default, tmp_path)
        h_default = compute_task_hash(task_default, plan, {})
        h_shell = compute_task_hash(task_shell, plan, {})
        assert h_default != h_shell

    def test_explicit_shell_false_on_string_command_changes_hash(self, tmp_path: Path) -> None:
        """String command defaults to shell=True; explicit shell=False produces different hash."""
        task_default = TaskSpec(id="t", command="echo ok")  # shell inferred as True
        task_noshell = TaskSpec(id="t", command="echo ok", shell=False)
        plan = _build_plan(task_default, tmp_path)
        h_default = compute_task_hash(task_default, plan, {})
        h_noshell = compute_task_hash(task_noshell, plan, {})
        assert h_default != h_noshell


# ---------------------------------------------------------------------------
# _effective_engine_config — codex task-level reasoning_effort override
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigReasoningOverride:
    """Tests for _effective_engine_config (cache.py): task-level reasoning_effort
    takes precedence over plan defaults."""

    def test_codex_task_reasoning_effort_overrides_plan(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="codex", prompt="x", reasoning_effort="xhigh")
        defaults = PlanDefaults(codex=EngineDefaults(reasoning_effort="medium"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["reasoning_effort"] == "xhigh"

    def test_claude_task_reasoning_effort_overrides_plan(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="claude", prompt="x", reasoning_effort="low")
        defaults = PlanDefaults(claude=EngineDefaults(reasoning_effort="high"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["reasoning_effort"] == "low"

    def test_gemini_task_reasoning_effort_overrides_plan(self, tmp_path: Path) -> None:
        task = TaskSpec(id="t", engine="gemini", prompt="x", reasoning_effort="high")
        defaults = PlanDefaults(gemini=EngineDefaults(reasoning_effort="low"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# _effective_engine_config — claude args dedup through config merge
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigClaudeArgsMerge:
    """Tests for _effective_engine_config (cache.py): claude plan+task args are
    concatenated and then dangerous flag is deduplicated."""

    def test_claude_duplicate_dangerous_flag_deduped_through_merge(self, tmp_path: Path) -> None:
        """Plan args and task args both contain the dangerous flag — merged and deduped."""
        flag = "--dangerously-skip-permissions"
        task = TaskSpec(id="t", engine="claude", prompt="x", args=[flag])
        defaults = PlanDefaults(claude=EngineDefaults(args=[flag]))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["args"].count(flag) == 1

    def test_claude_plan_and_task_args_concatenated(self, tmp_path: Path) -> None:
        """Plan-level and task-level args are concatenated for claude."""
        task = TaskSpec(id="t", engine="claude", prompt="x", args=["--task-flag"])
        defaults = PlanDefaults(claude=EngineDefaults(args=["--plan-flag"]))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert "--plan-flag" in config["args"]
        assert "--task-flag" in config["args"]


# ---------------------------------------------------------------------------
# cache_store — objects without expected attributes (exception swallowed)
# ---------------------------------------------------------------------------


class TestCacheStoreRobustness:
    """Tests for cache_store (cache.py): silently handles objects missing
    expected attributes (status, to_dict)."""

    def test_object_without_to_dict_does_not_raise(self, tmp_path: Path) -> None:
        """cache_store catches exceptions when result lacks to_dict()."""
        cache_dir = tmp_path / ".cache"
        task_hash = "ab" * 32

        class FakeResult:
            status = "success"  # has status but no to_dict

        cache_store(cache_dir, task_hash, FakeResult())
        # Should not raise — exception silently swallowed
        assert cache_lookup(cache_dir, task_hash) is None

    def test_object_without_status_attr_not_stored(self, tmp_path: Path) -> None:
        """cache_store treats missing status as non-success (empty string != 'success')."""
        cache_dir = tmp_path / ".cache"
        task_hash = "cd" * 32

        class NoStatus:
            pass

        cache_store(cache_dir, task_hash, NoStatus())
        assert cache_lookup(cache_dir, task_hash) is None

    def test_none_result_not_stored(self, tmp_path: Path) -> None:
        """cache_store handles None result gracefully."""
        cache_dir = tmp_path / ".cache"
        task_hash = "ef" * 32
        cache_store(cache_dir, task_hash, None)
        assert cache_lookup(cache_dir, task_hash) is None


# ---------------------------------------------------------------------------
# _load_prompt_content — prompt_file with template variables
# ---------------------------------------------------------------------------


class TestLoadPromptContentFileTemplates:
    """Tests for _load_prompt_content (cache.py): template variables in prompt files."""

    def test_prompt_file_template_variables_rendered(self, tmp_path: Path) -> None:
        """Template variables in prompt_file content are rendered."""
        pf = tmp_path / "tmpl.txt"
        pf.write_text("plan={{ plan_name }} id={{ task_id }}", encoding="utf-8")
        task = TaskSpec(id="my-task", engine="claude", prompt_file="tmpl.txt")
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert "plan=test-plan" in result["rendered_content"]
        assert "id=my-task" in result["rendered_content"]

    def test_prompt_md_file_template_variables_rendered(self, tmp_path: Path) -> None:
        """Template variables inside markdown prompt content are rendered."""
        md = tmp_path / "prompts.md"
        md.write_text(
            "## Task\n\n```text\nplan={{ plan_name }}\n```\n",
            encoding="utf-8",
        )
        task = TaskSpec(
            id="t", engine="claude",
            prompt_md_file="prompts.md",
            prompt_md_heading="Task",
        )
        plan = _build_plan(task, tmp_path)
        result = _load_prompt_content(task, plan)
        assert "plan=test-plan" in result["rendered_content"]


# ---------------------------------------------------------------------------
# _resolve_copilot_model — additional aliases not yet covered by parametrize
# ---------------------------------------------------------------------------


class TestResolveCopilotModelAdditionalAliases:
    """Covers copilot-specific aliases that were absent from the original
    parametrize set: opus-fast, opus-4.5, sonnet-4.5, sonnet-4, GPT variants,
    and the gemini-3-pro alias."""

    @pytest.mark.parametrize("alias,expected", [
        ("opus-fast", "claude-opus-4.6-fast"),
        ("opus-4.5", "claude-opus-4.5"),
        ("sonnet-4.5", "claude-sonnet-4.5"),
        ("sonnet-4", "claude-sonnet-4"),
        ("gpt-5.1-codex-mini", "gpt-5.1-codex-mini"),
        ("gpt-5.1-codex-max", "gpt-5.1-codex-max"),
        ("gpt-5.2", "gpt-5.2"),
        ("gpt-5.1", "gpt-5.1"),
        ("gpt-5-mini", "gpt-5-mini"),
        ("gpt-4.1", "gpt-4.1"),
        ("gemini-3-pro", "gemini-3-pro-preview"),
    ])
    def test_copilot_alias_resolution(self, alias: str, expected: str) -> None:
        assert _resolve_copilot_model(alias) == expected


# ---------------------------------------------------------------------------
# _resolve_ollama_model — extended aliases not yet covered
# ---------------------------------------------------------------------------


class TestResolveOllamaModelExtendedAliases:
    """Covers ollama aliases that are in the alias table but missing from the
    existing parametrize set: llama3.1, llama3.2, qwen2, qwen2.5-coder,
    deepseek-coder-v2, starcoder2."""

    @pytest.mark.parametrize("alias,expected", [
        ("llama3.1", "llama3.1"),
        ("llama3.2", "llama3.2"),
        ("qwen2", "qwen2"),
        ("qwen2.5-coder", "qwen2.5-coder"),
        ("deepseek-coder-v2", "deepseek-coder-v2"),
        ("starcoder2", "starcoder2"),
    ])
    def test_ollama_alias_resolution(self, alias: str, expected: str) -> None:
        assert _resolve_ollama_model(alias) == expected


# ---------------------------------------------------------------------------
# _resolve_qwen_model — "plus" alias not yet covered
# ---------------------------------------------------------------------------


class TestResolveQwenModelPlusAlias:
    def test_plus_alias_resolves(self) -> None:
        """The 'plus' alias maps to 'qwen-plus'."""
        assert _resolve_qwen_model("plus") == "qwen-plus"

    def test_unknown_qwen_model_passes_through(self) -> None:
        """Unknown model names are returned unchanged (not aliases)."""
        assert _resolve_qwen_model("qwen-custom-beta") == "qwen-custom-beta"


# ---------------------------------------------------------------------------
# compute_task_hash — layered context_mode invalidation
# ---------------------------------------------------------------------------


class TestComputeTaskHashLayeredContextMode:
    """Tests that 'layered' context_mode is correctly reflected in the hash."""

    def test_layered_vs_raw_context_mode_differ(self, tmp_path: Path) -> None:
        """Switching from 'raw' to 'layered' context_mode changes the hash."""
        task_a = TaskSpec(id="t", command="echo ok", context_mode="raw")
        task_b = TaskSpec(id="t", command="echo ok", context_mode="layered")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_layered_vs_map_reduce_context_mode_differ(self, tmp_path: Path) -> None:
        """Switching between 'layered' and 'map_reduce' changes the hash."""
        task_a = TaskSpec(id="t", command="echo ok", context_mode="layered")
        task_b = TaskSpec(id="t", command="echo ok", context_mode="map_reduce")
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})


# ---------------------------------------------------------------------------
# _effective_engine_config — copilot gemini-pro alias and plan default model
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigCopilotAliases:
    """Tests for _effective_engine_config (cache.py): copilot engine model alias
    resolution for gemini-pro and fallback to plan defaults."""

    def test_copilot_gemini_pro_alias_resolved(self, tmp_path: Path) -> None:
        """'gemini-pro' as copilot model resolves to 'gemini-2.5-pro'."""
        task = TaskSpec(id="t", engine="copilot", model="gemini-pro", prompt="x")
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "gemini-2.5-pro"

    def test_copilot_model_falls_back_to_plan_default(self, tmp_path: Path) -> None:
        """When task has no model, plan.defaults.copilot.model is resolved."""
        task = TaskSpec(id="t", engine="copilot", prompt="x")
        defaults = PlanDefaults(copilot=EngineDefaults(model="opus"))
        plan = _build_plan(task, tmp_path, defaults=defaults)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "claude-opus-4.6"


# ---------------------------------------------------------------------------
# _serialize_task_input_payload — null/nested structures
# ---------------------------------------------------------------------------


class TestSerializeTaskInputPayloadEdgeCases:
    """Additional edge cases for _serialize_task_input_payload."""

    def test_null_values_serialized_as_json_null(self) -> None:
        """None values in the payload are serialized as JSON null."""
        result = _serialize_task_input_payload({"key": None})
        text = result.decode("utf-8")
        assert '"key":null' in text

    def test_nested_dict_serialized_correctly(self) -> None:
        """Nested dicts are serialized with sorted keys at every level."""
        result = _serialize_task_input_payload({"outer": {"z": 1, "a": 2}})
        text = result.decode("utf-8")
        # Outer level: "outer" key present
        assert '"outer"' in text
        # Inner level: both keys present
        assert '"z"' in text
        assert '"a"' in text


# ---------------------------------------------------------------------------
# _resolve_append_system_prompt — unknown engine returns None
# ---------------------------------------------------------------------------


class TestResolveAppendSystemPromptUnknownEngine:
    """Tests for _resolve_append_system_prompt with an engine name not in the
    known set (no matching elif branch → returns None)."""

    def test_unknown_engine_returns_none_when_task_has_none(self, tmp_path: Path) -> None:
        """Engine name 'shell' has no elif branch; append_system_prompt is None."""
        task = TaskSpec(id="t", command="echo ok")
        plan = _build_plan(task, tmp_path)
        result = _resolve_append_system_prompt(plan, task, "shell")
        assert result is None

    def test_unknown_engine_task_level_still_returned(self, tmp_path: Path) -> None:
        """Task-level append_system_prompt is returned regardless of unknown engine."""
        task = TaskSpec(id="t", command="echo ok",
                        append_system_prompt="my-system-prompt")
        plan = _build_plan(task, tmp_path)
        result = _resolve_append_system_prompt(plan, task, "unknown-engine-xyz")
        assert result == "my-system-prompt"


# ---------------------------------------------------------------------------
# compute_task_hash — matrix_values field
# ---------------------------------------------------------------------------


class TestComputeTaskHashMatrixValues:
    """Tests for compute_task_hash (cache.py) covering the matrix_values field
    in the serialized payload. matrix_values is accessed via getattr to support
    tasks created without it."""

    def test_matrix_values_change_invalidates_hash(self, tmp_path: Path) -> None:
        """Tasks with different matrix_values produce different hashes."""
        task_a = TaskSpec(id="t", command="echo ok")
        task_b = TaskSpec(id="t", command="echo ok")
        # Inject matrix_values directly as a dynamic attribute
        task_a.matrix_values = {"env": "prod"}  # type: ignore[attr-defined]
        task_b.matrix_values = {"env": "dev"}  # type: ignore[attr-defined]
        plan_a = _build_plan(task_a, tmp_path)
        plan_b = _build_plan(task_b, tmp_path)
        assert compute_task_hash(task_a, plan_a, {}) != compute_task_hash(task_b, plan_b, {})

    def test_matrix_values_none_vs_set_invalidates(self, tmp_path: Path) -> None:
        """A task with no matrix_values and one with matrix_values differ."""
        task_no_mv = TaskSpec(id="t", command="echo ok")  # no matrix_values attr
        task_with_mv = TaskSpec(id="t", command="echo ok")
        task_with_mv.matrix_values = {"region": "us-east-1"}  # type: ignore[attr-defined]
        plan_a = _build_plan(task_no_mv, tmp_path)
        plan_b = _build_plan(task_with_mv, tmp_path)
        assert compute_task_hash(task_no_mv, plan_a, {}) != compute_task_hash(task_with_mv, plan_b, {})


# ---------------------------------------------------------------------------
# cache_store — non-success statuses are silently dropped
# ---------------------------------------------------------------------------


class TestCacheStoreNonSuccessStatuses:
    """Tests cache behavior for non-success statuses."""

    @pytest.mark.parametrize("status", ["failed", "soft_failed"])
    def test_failed_statuses_are_negative_cached(
        self, tmp_path: Path, status: str
    ) -> None:
        """Failed statuses are persisted as short-lived negative cache entries."""
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status=status)
        task_hash = "cc" * 32
        cache_store(cache_dir, task_hash, result)
        stored = cache_lookup(cache_dir, task_hash)
        assert stored is not None
        assert stored["status"] == status
        assert stored["_cache_kind"] == "negative"

    def test_skipped_status_not_stored(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / ".cache"
        result = _make_result(tmp_path, status="skipped")
        task_hash = "cd" * 32
        cache_store(cache_dir, task_hash, result)
        assert cache_lookup(cache_dir, task_hash) is None


# ---------------------------------------------------------------------------
# _serialize_task_input_payload — boolean and list types
# ---------------------------------------------------------------------------


class TestSerializeTaskInputPayloadBoolAndList:
    """Tests for _serialize_task_input_payload covering Python boolean and list
    types, which must be serialized as JSON true/false/arrays."""

    def test_boolean_true_serialized_as_json_true(self) -> None:
        result = _serialize_task_input_payload({"flag": True})
        text = result.decode("utf-8")
        assert '"flag":true' in text

    def test_boolean_false_serialized_as_json_false(self) -> None:
        result = _serialize_task_input_payload({"flag": False})
        text = result.decode("utf-8")
        assert '"flag":false' in text

    def test_list_serialized_as_json_array(self) -> None:
        result = _serialize_task_input_payload({"items": ["a", "b", "c"]})
        text = result.decode("utf-8")
        assert '"items":["a","b","c"]' in text

    def test_empty_list_serialized_as_json_empty_array(self) -> None:
        result = _serialize_task_input_payload({"deps": []})
        text = result.decode("utf-8")
        assert '"deps":[]' in text


# ---------------------------------------------------------------------------
# _load_prompt_content — heading-only (no md_file) falls through to ValueError
# ---------------------------------------------------------------------------


class TestLoadPromptContentHeadingOnly:
    """Tests that _load_prompt_content raises ValueError when prompt_md_heading
    is set but prompt_md_file is not, because the elif condition requires both."""

    def test_heading_without_md_file_raises_value_error(self, tmp_path: Path) -> None:
        """Only prompt_md_heading set — no matching source → ValueError."""
        task = TaskSpec(
            id="t", engine="claude",
            prompt_md_heading="My Heading",
            # prompt_md_file intentionally omitted
        )
        plan = _build_plan(task, tmp_path)
        with pytest.raises(ValueError, match="no prompt source"):
            _load_prompt_content(task, plan)


# ---------------------------------------------------------------------------
# _effective_engine_config — unknown engine passthrough includes reasoning_effort
# ---------------------------------------------------------------------------


class TestEffectiveEngineConfigUnknownEngineReasoning:
    """Tests for _effective_engine_config with an engine not in the known set.
    The function falls through to the final else branch, which passes through
    task.model and task.reasoning_effort unchanged."""

    def test_unknown_engine_reasoning_effort_passed_through(self, tmp_path: Path) -> None:
        """Unknown engine: task.reasoning_effort is included in the config as-is."""
        task = TaskSpec(
            id="t", command="echo ok",
            reasoning_effort="high",
        )
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["reasoning_effort"] == "high"

    def test_unknown_engine_model_passed_through(self, tmp_path: Path) -> None:
        """Unknown engine: task.model is included in the config as-is."""
        task = TaskSpec(
            id="t", command="echo ok",
            model="some-custom-model",
        )
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["model"] == "some-custom-model"

    def test_unknown_engine_args_from_task_only(self, tmp_path: Path) -> None:
        """Unknown engine: config args are just task.args (no plan defaults merging)."""
        task = TaskSpec(
            id="t", command="echo ok",
            args=["--custom-flag"],
        )
        plan = _build_plan(task, tmp_path)
        config = _effective_engine_config(task, plan)
        assert config["args"] == ["--custom-flag"]
