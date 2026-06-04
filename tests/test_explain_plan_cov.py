from __future__ import annotations

from pathlib import Path

import pytest

import maestro_cli.explain as explain_mod
from maestro_cli.explain import (
    PlanExplanation,
    _read_cache_stats,
    _short_hash,
    _topological_sort,
    explain_plan,
)
from maestro_cli.models import PlanSpec, TaskSpec


# ---------------------------------------------------------------------------
# _read_cache_stats
# ---------------------------------------------------------------------------


def test_read_cache_stats_none_returns_zero_tuple() -> None:
    """A None cache dir short-circuits to (0, 0) without touching cache_stats."""
    assert _read_cache_stats(None) == (0, 0)


def test_read_cache_stats_coerces_raw_values_to_ints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Non-None cache dir delegates to cache_stats and coerces fields to ints."""
    monkeypatch.setattr(
        explain_mod,
        "cache_stats",
        lambda _dir: {"entries": "7", "total_size_bytes": 4096.0},
    )
    assert _read_cache_stats(tmp_path) == (7, 4096)


def test_read_cache_stats_defaults_missing_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing keys in the raw stats dict default to zero."""
    monkeypatch.setattr(explain_mod, "cache_stats", lambda _dir: {})
    assert _read_cache_stats(tmp_path) == (0, 0)


# ---------------------------------------------------------------------------
# _short_hash
# ---------------------------------------------------------------------------


def test_short_hash_returns_value_when_short() -> None:
    """A value of 12 chars or fewer is returned unchanged."""
    assert _short_hash("abc123") == "abc123"
    # Exactly 12 chars is still <= 12 and returned untouched.
    assert _short_hash("0123456789ab") == "0123456789ab"


def test_short_hash_truncates_long_value() -> None:
    """A value longer than 12 chars is truncated with an ellipsis."""
    out = _short_hash("0123456789abcdef")
    assert out == "0123456789ab..."
    assert "..." in out


# ---------------------------------------------------------------------------
# _topological_sort
# ---------------------------------------------------------------------------


def test_topological_sort_orders_by_dependencies() -> None:
    """Dependents are emitted after the tasks they depend on (Kahn's algorithm)."""
    plan = PlanSpec(
        name="p",
        tasks=[
            TaskSpec(id="c", command="echo c", depends_on=["b"]),
            TaskSpec(id="b", command="echo b", depends_on=["a"]),
            TaskSpec(id="a", command="echo a"),
        ],
    )
    ordered = _topological_sort(plan)
    ids = [task.id for task in ordered]
    assert ids.index("a") < ids.index("b") < ids.index("c")
    assert len(ids) == 3


def test_topological_sort_fan_out_releases_multiple_dependents() -> None:
    """A single root with several dependents drives the dependents loop branch."""
    plan = PlanSpec(
        name="fan",
        tasks=[
            TaskSpec(id="root", command="echo root"),
            TaskSpec(id="left", command="echo left", depends_on=["root"]),
            TaskSpec(id="right", command="echo right", depends_on=["root"]),
        ],
    )
    ordered = _topological_sort(plan)
    ids = [task.id for task in ordered]
    assert ids[0] == "root"
    assert set(ids[1:]) == {"left", "right"}


def test_topological_sort_unknown_dependency_raises() -> None:
    """A dependency on a task that does not exist raises ValueError."""
    plan = PlanSpec(
        name="p",
        tasks=[TaskSpec(id="a", command="echo a", depends_on=["ghost"])],
    )
    with pytest.raises(ValueError, match="unknown task"):
        _topological_sort(plan)


def test_topological_sort_cycle_raises() -> None:
    """A dependency cycle leaves tasks unordered and raises ValueError."""
    plan = PlanSpec(
        name="p",
        tasks=[
            TaskSpec(id="a", command="echo a", depends_on=["b"]),
            TaskSpec(id="b", command="echo b", depends_on=["a"]),
        ],
    )
    with pytest.raises(ValueError, match="cycle"):
        _topological_sort(plan)


# ---------------------------------------------------------------------------
# explain_plan
# ---------------------------------------------------------------------------


def test_explain_plan_marks_cache_disabled_task(tmp_path: Path) -> None:
    """A task with cache=False is reported as 'disabled' and skips hashing."""
    plan = PlanSpec(
        name="disabled-plan",
        tasks=[TaskSpec(id="t1", command="echo hi", cache=False)],
    )
    result = explain_plan(plan, tmp_path)
    assert isinstance(result, PlanExplanation)
    assert result.plan_name == "disabled-plan"
    assert result.task_count == 1
    assert len(result.tasks) == 1
    item = result.tasks[0]
    assert item.task_id == "t1"
    assert item.cache_status == "disabled"
    assert "disabled" in item.reason


def test_explain_plan_no_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """With cache_dir=None, cache-enabled tasks are reported as 'no-cache-dir'."""
    plan = PlanSpec(
        name="nodir-plan",
        tasks=[TaskSpec(id="t1", command="echo hi")],
    )
    result = explain_plan(plan, None)
    assert result.cache_entries == 0
    assert result.cache_size_bytes == 0
    item = result.tasks[0]
    assert item.cache_status == "no-cache-dir"
    assert "no cache directory" in item.reason


def test_explain_plan_hash_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hash computation exception is captured as a 'miss' with a failure reason."""
    monkeypatch.setattr(
        explain_mod,
        "cache_stats",
        lambda _dir: {"entries": 0, "total_size_bytes": 0},
    )

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("hash boom")

    monkeypatch.setattr(explain_mod, "compute_task_hash", boom)

    plan = PlanSpec(
        name="hash-fail",
        tasks=[TaskSpec(id="t1", command="echo hi")],
    )
    result = explain_plan(plan, tmp_path)
    item = result.tasks[0]
    assert item.cache_status == "miss"
    assert "failed" in item.reason
    # No hash is recorded when computation fails.
    assert item.current_hash is None


def test_explain_plan_cache_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A computed hash with no cached entry is a 'miss' carrying the current hash."""
    monkeypatch.setattr(
        explain_mod,
        "cache_stats",
        lambda _dir: {"entries": 3, "total_size_bytes": 99},
    )
    monkeypatch.setattr(
        explain_mod, "compute_task_hash", lambda *a, **k: "hashforT1xyz"
    )
    monkeypatch.setattr(explain_mod, "cache_lookup", lambda _dir, _hash: None)

    plan = PlanSpec(
        name="miss-plan",
        tasks=[TaskSpec(id="t1", command="echo hi")],
    )
    result = explain_plan(plan, tmp_path)
    assert result.cache_entries == 3
    assert result.cache_size_bytes == 99
    item = result.tasks[0]
    assert item.cache_status == "miss"
    assert "no cached result" in item.reason
    assert item.current_hash == "hashforT1xyz"


def test_explain_plan_cache_hit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A computed hash with a cached entry is a 'hit' echoing both hash fields."""
    long_hash = "0123456789abcdefdeadbeef"
    monkeypatch.setattr(
        explain_mod,
        "cache_stats",
        lambda _dir: {"entries": 1, "total_size_bytes": 10},
    )
    monkeypatch.setattr(
        explain_mod, "compute_task_hash", lambda *a, **k: long_hash
    )
    monkeypatch.setattr(
        explain_mod, "cache_lookup", lambda _dir, _hash: {"status": "success"}
    )

    plan = PlanSpec(
        name="hit-plan",
        tasks=[TaskSpec(id="t1", command="echo hi")],
    )
    result = explain_plan(plan, tmp_path)
    item = result.tasks[0]
    assert item.cache_status == "hit"
    assert "hash match" in item.reason
    # The short hash (truncated) appears in the reason.
    assert "0123456789ab" in item.reason
    assert item.current_hash == long_hash
    assert item.cached_hash == long_hash


def test_explain_plan_chains_upstream_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful hash for an upstream task is fed into downstream hashing."""
    monkeypatch.setattr(
        explain_mod,
        "cache_stats",
        lambda _dir: {"entries": 0, "total_size_bytes": 0},
    )

    seen_upstreams: list[dict[str, str]] = []

    def fake_hash(task: TaskSpec, _plan: PlanSpec, upstream: dict[str, str]) -> str:
        # Record a copy of the upstream map visible to each task.
        seen_upstreams.append(dict(upstream))
        return f"hash-{task.id}"

    monkeypatch.setattr(explain_mod, "compute_task_hash", fake_hash)
    monkeypatch.setattr(explain_mod, "cache_lookup", lambda _dir, _hash: None)

    plan = PlanSpec(
        name="chain-plan",
        tasks=[
            TaskSpec(id="up", command="echo up"),
            TaskSpec(id="down", command="echo down", depends_on=["up"]),
        ],
    )
    result = explain_plan(plan, tmp_path)
    assert [item.task_id for item in result.tasks] == ["up", "down"]
    # First task saw an empty upstream map; second saw the first task's hash.
    assert seen_upstreams[0] == {}
    assert seen_upstreams[1] == {"up": "hash-up"}
