from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.cost_backfill import (
    _backfill_single_run,
    _coerce_cost,
    _infer_engine,
    backfill_run_costs,
    discover_run_roots,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_backfill_updates_manifest_and_result_file(tmp_path: Path) -> None:
    run_root = tmp_path / ".maestro-runs"
    run_dir = run_root / "run-1"
    run_dir.mkdir(parents=True)

    log_path = run_dir / "task-1.log"
    log_path.write_text('{"type":"result","total_cost_usd":1.75}\n', encoding="utf-8")

    result_path = run_dir / "task-1.result.json"
    _write_json(result_path, {"task_id": "task-1", "cost_usd": None})

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "run_id": "run-1",
            "task_results": {
                "task-1": {
                    "task_id": "task-1",
                    "log_path": str(log_path),
                    "result_path": str(result_path),
                    "cost_usd": None,
                }
            },
            "total_cost_usd": None,
        },
    )

    summary = backfill_run_costs(run_roots=[run_root], write=True)
    assert summary.runs_scanned == 1
    assert summary.runs_updated == 1
    assert summary.tasks_updated == 1
    assert summary.result_files_updated == 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-1"]["cost_usd"] == 1.75
    assert manifest["total_cost_usd"] == 1.75

    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_payload["cost_usd"] == 1.75


def test_backfill_dry_run_does_not_write_files(tmp_path: Path) -> None:
    run_root = tmp_path / ".maestro-runs"
    run_dir = run_root / "run-1"
    run_dir.mkdir(parents=True)

    log_path = run_dir / "task-1.log"
    log_path.write_text('{"type":"result","total_cost_usd":2.25}\n', encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "run_id": "run-1",
            "task_results": {
                "task-1": {
                    "task_id": "task-1",
                    "log_path": str(log_path),
                    "cost_usd": None,
                }
            },
            "total_cost_usd": None,
        },
    )

    summary = backfill_run_costs(run_roots=[run_root], write=False)
    assert summary.runs_scanned == 1
    assert summary.runs_updated == 1
    assert summary.tasks_updated == 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-1"]["cost_usd"] is None
    assert manifest["total_cost_usd"] is None


def test_discover_run_roots_finds_root_and_nested(tmp_path: Path) -> None:
    (tmp_path / ".maestro-runs").mkdir()
    (tmp_path / "plans" / ".maestro-runs").mkdir(parents=True)
    (tmp_path / "clients" / "alpha" / "plans" / ".maestro-runs").mkdir(parents=True)
    (tmp_path / "docs").mkdir()

    roots = discover_run_roots(tmp_path)
    assert (tmp_path / ".maestro-runs") in roots
    assert (tmp_path / "plans" / ".maestro-runs") in roots
    assert (tmp_path / "clients" / "alpha" / "plans" / ".maestro-runs") in roots


# ---------------------------------------------------------------------------
# _coerce_cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1.75", 1.75),
        (2, 2.0),
        (0, 0.0),
        ("0.0", 0.0),
        (-0.5, None),
        ("-1", None),
        (None, None),
        ("not-a-number", None),
        ("", None),
    ],
)
def test_coerce_cost_variants(value: object, expected: float | None) -> None:
    from maestro_cli.cost_backfill import _coerce_cost

    assert _coerce_cost(value) == expected


# ---------------------------------------------------------------------------
# _infer_engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        ("codex exec --prompt 'hello'", "codex"),
        ("codex", "codex"),
        ("claude --print 'hi'", "claude"),
        ("gemini -m flash -p 'hi'", "gemini"),
        ("copilot --autopilot -p 'hi'", "copilot"),
        ("bash -c 'echo hi'", None),
        ("", None),
        ("python script.py", None),
    ],
)
def test_infer_engine_variants(command: str, expected: str | None) -> None:
    assert _infer_engine(command) == expected


# ---------------------------------------------------------------------------
# discover_run_roots — edge cases
# ---------------------------------------------------------------------------


def test_discover_run_roots_nonexistent_returns_empty(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert discover_run_roots(missing) == []


def test_discover_run_roots_skips_hidden_and_skip_dirs(tmp_path: Path) -> None:
    # .git, node_modules, __pycache__ should be skipped
    (tmp_path / ".git" / ".maestro-runs").mkdir(parents=True)
    (tmp_path / "node_modules" / ".maestro-runs").mkdir(parents=True)
    (tmp_path / "__pycache__" / ".maestro-runs").mkdir(parents=True)
    # Only this one should be discovered
    (tmp_path / "src" / ".maestro-runs").mkdir(parents=True)

    roots = discover_run_roots(tmp_path)
    names = [r.parent.name for r in roots]
    assert "src" in names
    assert ".git" not in names
    assert "node_modules" not in names
    assert "__pycache__" not in names


def test_discover_run_roots_respects_max_depth(tmp_path: Path) -> None:
    # Place .maestro-runs 3 levels deep
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / ".maestro-runs").mkdir()

    # max_depth=2 should NOT find it (a=1, b=2, c=3 > max_depth)
    roots_shallow = discover_run_roots(tmp_path, max_depth=2)
    assert len(roots_shallow) == 0

    # max_depth=3 should find it
    roots_deep = discover_run_roots(tmp_path, max_depth=3)
    assert len(roots_deep) == 1


# ---------------------------------------------------------------------------
# _backfill_single_run — edge cases
# ---------------------------------------------------------------------------


def test_backfill_single_run_missing_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    changed, tasks, results = _backfill_single_run(run_dir, write=True)
    assert changed is False
    assert tasks == 0
    assert results == 0


def test_backfill_single_run_invalid_json_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("not valid json{{{", encoding="utf-8")
    changed, tasks, results = _backfill_single_run(run_dir, write=True)
    assert changed is False


def test_backfill_single_run_task_already_has_cost_and_tokens(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text('{"type":"result","total_cost_usd":9.99}\n', encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": 1.0,
                    "token_usage": {"total_tokens": 100},
                    "log_path": str(log_path),
                }
            },
            "total_cost_usd": 1.0,
        },
    )

    # Task already populated — nothing should change
    changed, tasks, results = _backfill_single_run(run_dir, write=True)
    assert tasks == 0
    # manifest unchanged (total_cost already matches)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-1"]["cost_usd"] == 1.0


def test_backfill_single_run_no_log_file_skips_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    # no log_path, no task-1.log on disk
                }
            },
            "total_cost_usd": None,
        },
    )

    changed, tasks, results = _backfill_single_run(run_dir, write=True)
    assert tasks == 0


def test_backfill_single_run_result_file_invalid_json_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Result file with invalid JSON is silently skipped; manifest is still updated."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    result_path = run_dir / "task-1.result.json"
    result_path.write_text("INVALID JSON", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "log_path": str(log_path),
                    "result_path": str(result_path),
                    "command": "claude --print hi",
                }
            },
            "total_cost_usd": None,
        },
    )

    # Monkeypatch extractors to return a known cost
    from maestro_cli.models import TokenUsage

    class _FakeCostResult:
        cost_usd = 0.42
        token_usage = None

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeCostResult())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks, results = _backfill_single_run(run_dir, write=True)
    # Cost was updated in manifest even though result file was bad
    assert tasks == 1
    assert results == 0  # result file not updated (bad JSON)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-1"]["cost_usd"] == 0.42


def test_backfill_run_costs_nonexistent_run_root_skipped(tmp_path: Path) -> None:
    missing = tmp_path / "ghost-root"
    summary = backfill_run_costs(run_roots=[missing], write=False)
    assert summary.runs_scanned == 0
    assert summary.run_roots == 1


# ---------------------------------------------------------------------------
# _resolve_task_log_path
# ---------------------------------------------------------------------------


def test_resolve_task_log_path_falls_back_to_task_id_log(tmp_path: Path) -> None:
    from maestro_cli.cost_backfill import _resolve_task_log_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    # No log_path in dict; create <task_id>.log as fallback
    fallback = run_dir / "my-task.log"
    fallback.write_text("log content", encoding="utf-8")

    result = _resolve_task_log_path(run_dir, "my-task", {})
    assert result == fallback


def test_resolve_task_log_path_returns_none_when_nothing_exists(tmp_path: Path) -> None:
    from maestro_cli.cost_backfill import _resolve_task_log_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    # Neither explicit log_path nor fallback file exist
    result = _resolve_task_log_path(run_dir, "missing-task", {})
    assert result is None


# ---------------------------------------------------------------------------
# _backfill_single_run — missing task_results key
# ---------------------------------------------------------------------------


def test_backfill_single_run_missing_task_results_key(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")

    changed, tasks, results = _backfill_single_run(run_dir, write=True)
    assert changed is False
    assert tasks == 0
    assert results == 0


# ---------------------------------------------------------------------------
# discover_run_roots — file as project root returns empty
# ---------------------------------------------------------------------------


def test_discover_run_roots_file_as_root_returns_empty(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-dir.txt"
    file_path.write_text("hello", encoding="utf-8")

    assert discover_run_roots(file_path) == []


# ---------------------------------------------------------------------------
# _resolve_result_path — direct tests
# ---------------------------------------------------------------------------


def test_resolve_result_path_no_key_returns_default(tmp_path: Path) -> None:
    from maestro_cli.cost_backfill import _resolve_result_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    result = _resolve_result_path(run_dir, "my-task", {})
    assert result == run_dir / "my-task.result.json"


def test_resolve_result_path_explicit_relative_path(tmp_path: Path) -> None:
    from maestro_cli.cost_backfill import _resolve_result_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    result = _resolve_result_path(run_dir, "my-task", {"result_path": "artifacts/my-task.result.json"})
    assert result == run_dir / "artifacts" / "my-task.result.json"


# ---------------------------------------------------------------------------
# _backfill_single_run — task has cost but missing token_usage
# ---------------------------------------------------------------------------


def test_backfill_single_run_task_has_cost_missing_tokens_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When cost is present but token_usage is absent, tokens should be backfilled."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": 0.5,  # cost already present
                    "token_usage": None,  # tokens missing
                    "log_path": str(log_path),
                    "command": "claude --print hi",
                }
            },
            "total_cost_usd": 0.5,
        },
    )

    from maestro_cli.models import TokenUsage

    class _FakeCostResult:
        cost_usd = 0.5
        token_usage = TokenUsage(input_tokens=100, output_tokens=50)

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeCostResult())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks_updated, _results = _backfill_single_run(run_dir, write=True)

    assert changed is True
    assert tasks_updated == 0  # cost was already present, so tasks_updated not incremented
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_usage = manifest["task_results"]["task-1"]["token_usage"]
    assert token_usage is not None
    assert token_usage["total_tokens"] == 150  # input(100) + output(50)


# ---------------------------------------------------------------------------
# backfill_run_costs — multiple run roots
# ---------------------------------------------------------------------------


def test_backfill_run_costs_multiple_run_roots(tmp_path: Path) -> None:
    """Valid run root and a non-existent root — only the valid one is scanned."""
    run_root_a = tmp_path / "project-a" / ".maestro-runs"
    run_dir_a = run_root_a / "run-1"
    run_dir_a.mkdir(parents=True)

    log_path = run_dir_a / "task-1.log"
    log_path.write_text('{"type":"result","total_cost_usd":1.0}\n', encoding="utf-8")

    _write_json(
        run_dir_a / "run_manifest.json",
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "log_path": str(log_path),
                }
            },
            "total_cost_usd": None,
        },
    )

    missing_root = tmp_path / "ghost" / ".maestro-runs"

    summary = backfill_run_costs(run_roots=[run_root_a, missing_root], write=False)
    assert summary.run_roots == 2
    assert summary.runs_scanned == 1  # only run_root_a has runs
    assert summary.runs_updated == 1


# ---------------------------------------------------------------------------
# _resolve_task_log_path — absolute log_path that exists
# ---------------------------------------------------------------------------


def test_resolve_task_log_path_absolute_exists(tmp_path: Path) -> None:
    from maestro_cli.cost_backfill import _resolve_task_log_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    absolute_log = tmp_path / "elsewhere" / "my-task.log"
    absolute_log.parent.mkdir(parents=True)
    absolute_log.write_text("log content", encoding="utf-8")

    result = _resolve_task_log_path(run_dir, "my-task", {"log_path": str(absolute_log)})
    assert result == absolute_log


# ---------------------------------------------------------------------------
# _backfill_single_run — total_tokens aggregated into manifest
# ---------------------------------------------------------------------------


def test_backfill_single_run_aggregates_total_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After backfill, manifest.total_tokens is set to sum of updated task token counts."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "token_usage": None,
                    "log_path": str(log_path),
                    "command": "claude --print hi",
                }
            },
            "total_cost_usd": None,
            "total_tokens": None,
        },
    )

    from maestro_cli.models import TokenUsage

    class _FakeCostResult:
        cost_usd = 0.10
        token_usage = TokenUsage(input_tokens=200, output_tokens=100)

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeCostResult())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks_updated, _results = _backfill_single_run(run_dir, write=True)

    assert changed is True
    assert tasks_updated == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_tokens"] == 300  # input(200) + output(100)


# ---------------------------------------------------------------------------
# _resolve_task_log_path — relative log_path resolved against run_dir
# ---------------------------------------------------------------------------


def test_resolve_task_log_path_relative_path_exists(tmp_path: Path) -> None:
    from maestro_cli.cost_backfill import _resolve_task_log_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    relative_log = run_dir / "logs" / "my-task.log"
    relative_log.parent.mkdir(parents=True)
    relative_log.write_text("log content", encoding="utf-8")

    result = _resolve_task_log_path(run_dir, "my-task", {"log_path": "logs/my-task.log"})
    assert result == relative_log


# ---------------------------------------------------------------------------
# _resolve_result_path — absolute path in dict returned directly
# ---------------------------------------------------------------------------


def test_resolve_result_path_absolute_path(tmp_path: Path) -> None:
    from maestro_cli.cost_backfill import _resolve_result_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    absolute_result = tmp_path / "elsewhere" / "my-task.result.json"
    result = _resolve_result_path(run_dir, "my-task", {"result_path": str(absolute_result)})
    assert result == absolute_result


# ---------------------------------------------------------------------------
# _backfill_single_run — multiple tasks, only some need updating
# ---------------------------------------------------------------------------


def test_backfill_single_run_multiple_tasks_partial_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One task already fully populated is skipped; another missing cost is updated."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path_a = run_dir / "task-a.log"
    log_path_a.write_text("", encoding="utf-8")
    log_path_b = run_dir / "task-b.log"
    log_path_b.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-a": {
                    "cost_usd": 1.0,
                    "token_usage": {"total_tokens": 100},
                    "log_path": str(log_path_a),
                    "command": "claude --print hi",
                },
                "task-b": {
                    "cost_usd": None,
                    "token_usage": None,
                    "log_path": str(log_path_b),
                    "command": "claude --print hi",
                },
            },
            "total_cost_usd": 1.0,
        },
    )

    from maestro_cli.models import TokenUsage

    class _FakeCostResult:
        cost_usd = 0.20
        token_usage = TokenUsage(input_tokens=50, output_tokens=50)

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeCostResult())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks_updated, _results = _backfill_single_run(run_dir, write=True)

    assert changed is True
    assert tasks_updated == 1  # only task-b was updated
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-a"]["cost_usd"] == 1.0  # unchanged
    assert manifest["task_results"]["task-b"]["cost_usd"] == 0.20  # updated
    assert manifest["total_cost_usd"] == pytest.approx(1.20)


# ---------------------------------------------------------------------------
# backfill_run_costs — manifests_failed incremented on exception
# ---------------------------------------------------------------------------


def test_backfill_run_costs_manifests_failed_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When _backfill_single_run raises, manifests_failed is incremented."""
    import maestro_cli.cost_backfill as mod

    run_root = tmp_path / ".maestro-runs"
    run_dir = run_root / "run-1"
    run_dir.mkdir(parents=True)
    # Create a manifest so the run dir is scanned
    (run_dir / "run_manifest.json").write_text(json.dumps({"task_results": {}}), encoding="utf-8")

    monkeypatch.setattr(mod, "_backfill_single_run", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    summary = backfill_run_costs(run_roots=[run_root], write=False)
    assert summary.runs_scanned == 1
    assert summary.manifests_failed == 1
    assert summary.runs_updated == 0


# ---------------------------------------------------------------------------
# _backfill_single_run — write=False returns changed but does not modify disk
# ---------------------------------------------------------------------------


def test_backfill_single_run_write_false_no_disk_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_backfill_single_run with write=False reports changed=True but leaves the file untouched."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    original_payload = {
        "task_results": {
            "task-1": {
                "cost_usd": None,
                "token_usage": None,
                "log_path": str(log_path),
                "command": "claude --print hi",
            }
        },
        "total_cost_usd": None,
    }
    _write_json(manifest_path, original_payload)

    class _FakeCostResult:
        cost_usd = 0.77
        token_usage = None

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeCostResult())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks_updated, _ = _backfill_single_run(run_dir, write=False)

    assert changed is True
    assert tasks_updated == 1
    # Manifest on disk must NOT have been modified
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["task_results"]["task-1"]["cost_usd"] is None
    assert on_disk["total_cost_usd"] is None


# ---------------------------------------------------------------------------
# _resolve_task_log_path — relative path that does not exist falls back
# ---------------------------------------------------------------------------


def test_resolve_task_log_path_relative_not_exists_uses_fallback(tmp_path: Path) -> None:
    from maestro_cli.cost_backfill import _resolve_task_log_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    # The relative log_path points nowhere
    # But the fallback <task_id>.log does exist
    fallback = run_dir / "my-task.log"
    fallback.write_text("fallback log", encoding="utf-8")

    result = _resolve_task_log_path(run_dir, "my-task", {"log_path": "nonexistent/path.log"})
    assert result == fallback


# ---------------------------------------------------------------------------
# _backfill_single_run — task_results value is not a dict (e.g., a list)
# ---------------------------------------------------------------------------


def test_backfill_single_run_task_results_not_dict_returns_false(tmp_path: Path) -> None:
    """When task_results is a list instead of a dict, the function returns (False, 0, 0)."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"run_id": "run-1", "task_results": ["not", "a", "dict"]}),
        encoding="utf-8",
    )

    changed, tasks, results = _backfill_single_run(run_dir, write=True)
    assert changed is False
    assert tasks == 0
    assert results == 0


# ---------------------------------------------------------------------------
# _backfill_single_run — no engine inferred, _extract_cost_from_log used
# ---------------------------------------------------------------------------


def test_backfill_single_run_no_engine_uses_extract_cost_from_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When command has no known engine, cost_and_tokens is None;
    _extract_cost_from_log is used as fallback."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "token_usage": None,
                    "log_path": str(log_path),
                    "command": "bash -c 'echo hello'",  # no known engine
                }
            },
            "total_cost_usd": None,
        },
    )

    # _extract_cost_and_tokens_from_log is never called for unknown engine;
    # _extract_cost_from_log is the fallback
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: 0.33)

    changed, tasks_updated, _ = _backfill_single_run(run_dir, write=True)

    assert changed is True
    assert tasks_updated == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-1"]["cost_usd"] == 0.33


# ---------------------------------------------------------------------------
# discover_run_roots — skips `venv` directory (in _DISCOVERY_SKIP_DIRS)
# ---------------------------------------------------------------------------


def test_discover_run_roots_skips_venv_directory(tmp_path: Path) -> None:
    """venv is in _DISCOVERY_SKIP_DIRS and must not be descended into."""
    (tmp_path / "venv" / ".maestro-runs").mkdir(parents=True)
    (tmp_path / "src" / ".maestro-runs").mkdir(parents=True)

    roots = discover_run_roots(tmp_path)
    parent_names = [r.parent.name for r in roots]
    assert "src" in parent_names
    assert "venv" not in parent_names


# ---------------------------------------------------------------------------
# _resolve_task_log_path — absolute path that does NOT exist falls back
# ---------------------------------------------------------------------------


def test_resolve_task_log_path_absolute_not_exists_falls_to_fallback(tmp_path: Path) -> None:
    """If an absolute log_path is given but the file does not exist,
    _resolve_task_log_path must fall through to the <task_id>.log fallback."""
    from maestro_cli.cost_backfill import _resolve_task_log_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    # Create the fallback file
    fallback = run_dir / "my-task.log"
    fallback.write_text("fallback", encoding="utf-8")

    # Provide an absolute path that does NOT exist
    ghost = tmp_path / "ghost" / "my-task.log"

    result = _resolve_task_log_path(run_dir, "my-task", {"log_path": str(ghost)})
    assert result == fallback


# ---------------------------------------------------------------------------
# backfill_run_costs — non-dir items in run_root are skipped
# ---------------------------------------------------------------------------


def test_backfill_run_costs_skips_nondirs_in_run_root(tmp_path: Path) -> None:
    """Files inside a run_root are skipped; only directories are treated as run dirs."""
    run_root = tmp_path / ".maestro-runs"
    run_root.mkdir()

    # Create a regular file (not a run dir) inside the run_root
    (run_root / "stale_lock.txt").write_text("lock", encoding="utf-8")

    summary = backfill_run_costs(run_roots=[run_root], write=False)
    assert summary.runs_scanned == 0


# ---------------------------------------------------------------------------
# _backfill_single_run — engine known but cost_and_tokens.cost_usd is None,
# fallback _extract_cost_from_log still provides the cost
# ---------------------------------------------------------------------------


def test_backfill_single_run_engine_cost_none_falls_back_to_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When engine is known but cost_and_tokens returns cost_usd=None,
    the fallback _extract_cost_from_log is consulted."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "token_usage": None,
                    "log_path": str(log_path),
                    "command": "claude --print hi",
                }
            },
            "total_cost_usd": None,
        },
    )

    class _FakeCostResultNoCost:
        cost_usd = None  # extractor yields no cost
        token_usage = None

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeCostResultNoCost())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: 0.55)

    changed, tasks_updated, _ = _backfill_single_run(run_dir, write=True)

    assert changed is True
    assert tasks_updated == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-1"]["cost_usd"] == 0.55


# ---------------------------------------------------------------------------
# _backfill_single_run — result file gets both cost AND token_usage updated
# ---------------------------------------------------------------------------


def test_backfill_single_run_result_file_updated_with_cost_and_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a result file has neither cost_usd nor token_usage, both are written
    from task_result after backfill, and result_files_updated is 1."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    result_path = run_dir / "task-1.result.json"
    _write_json(result_path, {"task_id": "task-1", "cost_usd": None, "token_usage": None})

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "token_usage": None,
                    "log_path": str(log_path),
                    "result_path": str(result_path),
                    "command": "claude --print hi",
                }
            },
            "total_cost_usd": None,
        },
    )

    from maestro_cli.models import TokenUsage

    class _FakeCostResult:
        cost_usd = 0.30
        token_usage = TokenUsage(input_tokens=150, output_tokens=75)

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeCostResult())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks_updated, result_files_updated = _backfill_single_run(run_dir, write=True)

    assert changed is True
    assert tasks_updated == 1
    assert result_files_updated == 1

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["cost_usd"] == 0.30
    assert payload["token_usage"] is not None
    assert payload["token_usage"]["total_tokens"] == 225  # 150 + 75


# ---------------------------------------------------------------------------
# _resolve_result_path — empty string result_path falls back to default
# ---------------------------------------------------------------------------


def test_resolve_result_path_empty_string_falls_back_to_default(tmp_path: Path) -> None:
    """An empty-string result_path is treated as absent; default <task_id>.result.json is returned."""
    from maestro_cli.cost_backfill import _resolve_result_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    result = _resolve_result_path(run_dir, "my-task", {"result_path": "   "})
    assert result == run_dir / "my-task.result.json"


# ---------------------------------------------------------------------------
# backfill_run_costs — empty run_root (no subdirs) yields runs_scanned == 0
# ---------------------------------------------------------------------------


def test_backfill_run_costs_empty_run_root_yields_zero_scanned(tmp_path: Path) -> None:
    """A run_root that exists but has no subdirectories produces runs_scanned == 0."""
    run_root = tmp_path / ".maestro-runs"
    run_root.mkdir()

    summary = backfill_run_costs(run_roots=[run_root], write=False)
    assert summary.run_roots == 1
    assert summary.runs_scanned == 0
    assert summary.runs_updated == 0


# ---------------------------------------------------------------------------
# _backfill_single_run — task_results value that is not a dict is skipped
# ---------------------------------------------------------------------------


def test_backfill_single_run_task_result_value_not_dict_skipped(tmp_path: Path) -> None:
    """When a task_results entry's value is not a dict (e.g., a string), it is silently skipped."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "task_results": {
                "task-bad": "this should be a dict but is not",
            },
            "total_cost_usd": None,
        }),
        encoding="utf-8",
    )

    changed, tasks, results = _backfill_single_run(run_dir, write=True)
    assert changed is False
    assert tasks == 0
    assert results == 0


# ---------------------------------------------------------------------------
# _resolve_result_path — explicit None value falls back to default
# ---------------------------------------------------------------------------


def test_resolve_result_path_none_value_falls_back_to_default(tmp_path: Path) -> None:
    """An explicit None result_path in the dict is treated as absent; default returned."""
    from maestro_cli.cost_backfill import _resolve_result_path

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    result = _resolve_result_path(run_dir, "my-task", {"result_path": None})
    assert result == run_dir / "my-task.result.json"


# ---------------------------------------------------------------------------
# _backfill_single_run — zero total_tokens not counted in total_tokens sum
# ---------------------------------------------------------------------------


def test_backfill_single_run_zero_total_tokens_excluded_from_sum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task whose token_usage has total_tokens == 0 must NOT be included in
    the manifest total_tokens aggregation (only > 0 counts)."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "token_usage": None,
                    "log_path": str(log_path),
                    "command": "claude --print hi",
                }
            },
            "total_cost_usd": None,
            "total_tokens": None,
        },
    )

    from maestro_cli.models import TokenUsage

    class _FakeZeroTokens:
        cost_usd = 0.05
        token_usage = TokenUsage(input_tokens=0, output_tokens=0)  # total_tokens == 0

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeZeroTokens())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks_updated, _ = _backfill_single_run(run_dir, write=True)

    assert changed is True
    assert tasks_updated == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # total_tokens must remain None because 0 is excluded from the sum
    assert manifest["total_tokens"] is None


# ---------------------------------------------------------------------------
# discover_run_roots — OSError during iterdir is handled gracefully
# ---------------------------------------------------------------------------


def test_discover_run_roots_handles_oserror_on_iterdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When iterdir raises OSError on a subdirectory, the walker skips it and continues."""
    # A valid .maestro-runs at root level should still be found
    (tmp_path / ".maestro-runs").mkdir()

    # Create a subdir that will trigger OSError
    bad_dir = tmp_path / "bad-dir"
    bad_dir.mkdir()

    original_iterdir = Path.iterdir

    def _fake_iterdir(self: Path):  # type: ignore[override]
        if self == bad_dir:
            raise OSError("permission denied")
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _fake_iterdir)

    roots = discover_run_roots(tmp_path)
    # The root-level .maestro-runs must still be discovered
    assert any(r.name == ".maestro-runs" for r in roots)


# ---------------------------------------------------------------------------
# discover_run_roots — max_depth=0 still finds root-level .maestro-runs
# ---------------------------------------------------------------------------


def test_discover_run_roots_max_depth_zero_finds_root_only(tmp_path: Path) -> None:
    """With max_depth=0, the root-level .maestro-runs is still found
    because it is checked before the depth guard, but nested ones are not."""
    (tmp_path / ".maestro-runs").mkdir()
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / ".maestro-runs").mkdir()

    roots = discover_run_roots(tmp_path, max_depth=0)
    assert len(roots) == 1
    assert roots[0].parent == tmp_path.resolve()


# ---------------------------------------------------------------------------
# backfill_run_costs — empty run_roots list
# ---------------------------------------------------------------------------


def test_backfill_run_costs_empty_run_roots(tmp_path: Path) -> None:
    """Calling backfill_run_costs with no run_roots produces a zeroed summary."""
    summary = backfill_run_costs(run_roots=[], write=False)
    assert summary.run_roots == 0
    assert summary.runs_scanned == 0
    assert summary.runs_updated == 0
    assert summary.tasks_updated == 0
    assert summary.manifests_failed == 0


# ---------------------------------------------------------------------------
# _backfill_single_run — result file already has cost, only token_usage synced
# ---------------------------------------------------------------------------


def test_backfill_single_run_result_file_already_has_cost_only_tokens_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the result file already has cost_usd set but lacks token_usage,
    only token_usage is written to the result file (cost left untouched)."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    result_path = run_dir / "task-1.result.json"
    _write_json(result_path, {"task_id": "task-1", "cost_usd": 0.50, "token_usage": None})

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,  # manifest lacks cost
                    "token_usage": None,
                    "log_path": str(log_path),
                    "result_path": str(result_path),
                    "command": "claude --print hi",
                }
            },
            "total_cost_usd": None,
        },
    )

    from maestro_cli.models import TokenUsage

    class _FakeCostResult:
        cost_usd = 0.50
        token_usage = TokenUsage(input_tokens=80, output_tokens=40)

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeCostResult())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    _changed, _tasks, result_files_updated = _backfill_single_run(run_dir, write=True)

    assert result_files_updated == 1
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    # cost_usd was already present — must not change
    assert payload["cost_usd"] == 0.50
    # token_usage was absent — must now be populated
    assert payload["token_usage"] is not None
    assert payload["token_usage"]["total_tokens"] == 120  # 80 + 40


# ---------------------------------------------------------------------------
# _backfill_single_run — both extractors return nothing → no update
# ---------------------------------------------------------------------------


def test_backfill_single_run_no_cost_extracted_no_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When engine is detected but both extractors return no cost or tokens,
    the manifest is not changed and tasks_updated stays 0."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "token_usage": None,
                    "log_path": str(log_path),
                    "command": "claude --print hi",
                }
            },
            "total_cost_usd": None,
        },
    )

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks_updated, result_files_updated = _backfill_single_run(run_dir, write=True)

    assert tasks_updated == 0
    assert result_files_updated == 0
    # manifest may still change (total_cost_usd None→None transition is a no-op)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-1"]["cost_usd"] is None


# ---------------------------------------------------------------------------
# _backfill_single_run — engine returns cost but token_usage is None
# ---------------------------------------------------------------------------


def test_backfill_single_run_cost_extracted_token_usage_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the combined extractor returns a cost but token_usage is None,
    cost_usd is updated in the manifest but token_usage is left unchanged."""
    import maestro_cli.cost_backfill as mod

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()

    log_path = run_dir / "task-1.log"
    log_path.write_text("", encoding="utf-8")

    manifest_path = run_dir / "run_manifest.json"
    _write_json(
        manifest_path,
        {
            "task_results": {
                "task-1": {
                    "cost_usd": None,
                    "token_usage": None,
                    "log_path": str(log_path),
                    "command": "codex exec hi",
                }
            },
            "total_cost_usd": None,
        },
    )

    class _FakeNoTokens:
        cost_usd = 0.33
        token_usage = None

    monkeypatch.setattr(mod, "_extract_cost_and_tokens_from_log", lambda *a, **kw: _FakeNoTokens())
    monkeypatch.setattr(mod, "_extract_cost_from_log", lambda *a, **kw: None)

    changed, tasks_updated, _result_files = _backfill_single_run(run_dir, write=True)

    assert changed is True
    assert tasks_updated == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["task_results"]["task-1"]["cost_usd"] == 0.33
    assert manifest["task_results"]["task-1"]["token_usage"] is None


# ---------------------------------------------------------------------------
# discover_run_roots — .maestro-runs is a file, not a directory → skipped
# ---------------------------------------------------------------------------


def test_discover_run_roots_maestro_runs_file_not_dir_skipped(tmp_path: Path) -> None:
    """A file named .maestro-runs is not treated as a run root."""
    # Create a *file* named .maestro-runs at the project root
    fake = tmp_path / ".maestro-runs"
    fake.write_text("not a directory", encoding="utf-8")

    roots = discover_run_roots(tmp_path)
    assert roots == []
