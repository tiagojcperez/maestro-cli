from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="web extras not installed")
pytest.importorskip("starlette", reason="web extras not installed")

from starlette.testclient import TestClient

from maestro_cli.web import create_app
from maestro_cli.web.state import _active_runs, _lock, set_project_roots


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(tmp_path: Path) -> None:
    """Reset global web state before and after each test."""
    set_project_roots([tmp_path])
    with _lock:
        _active_runs.clear()
    yield  # type: ignore[misc]
    set_project_roots([Path(".")])
    with _lock:
        _active_runs.clear()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(project_roots=[tmp_path])
    return TestClient(app, raise_server_exceptions=False)


def _write_manifest(
    root: Path,
    run_id: str,
    manifest: dict[str, Any],
) -> Path:
    """Write a run manifest JSON under root/.maestro-runs/<run_id>/."""
    run_dir = root / ".maestro-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def _make_manifest(
    plan_name: str = "test-plan",
    run_id: str = "run-001",
    success: bool = True,
    total_tokens: int | None = None,
    total_cost_usd: float | None = None,
    task_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "plan_name": plan_name,
        "run_id": run_id,
        "started_at": "2026-03-01T10:00:00+00:00",
        "finished_at": "2026-03-01T10:03:00+00:00",
        "success": success,
        "task_results": task_results or {},
    }
    if total_tokens is not None:
        manifest["total_tokens"] = total_tokens
    if total_cost_usd is not None:
        manifest["total_cost_usd"] = total_cost_usd
    return manifest


# ---------------------------------------------------------------------------
# Tests: GET /api/runs/stats — token aggregation
# ---------------------------------------------------------------------------


class TestStatsTokenFields:
    def test_stats_includes_total_tokens_key(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The stats response always includes a total_tokens key."""
        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_tokens" in data

    def test_stats_total_tokens_none_when_no_runs(
        self, client: TestClient
    ) -> None:
        """total_tokens is null when there are no run manifests."""
        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tokens"] is None

    def test_stats_total_tokens_aggregated_from_manifest(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """total_tokens is summed from manifests that carry total_tokens."""
        _write_manifest(tmp_path, "run-a", _make_manifest(total_tokens=1000))
        _write_manifest(tmp_path, "run-b", _make_manifest(total_tokens=500))

        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tokens"] == 1500

    def test_stats_avg_tokens_per_run(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """avg_tokens_per_run is the mean across runs that have token data."""
        _write_manifest(tmp_path, "run-a", _make_manifest(total_tokens=2000))
        _write_manifest(tmp_path, "run-b", _make_manifest(total_tokens=1000))

        resp = client.get("/api/runs/stats")
        data = resp.json()
        assert data["avg_tokens_per_run"] == 1500

    def test_stats_avg_tokens_none_when_no_token_data(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """avg_tokens_per_run is null when no manifests carry token data."""
        _write_manifest(tmp_path, "run-a", _make_manifest())  # no total_tokens

        resp = client.get("/api/runs/stats")
        data = resp.json()
        assert data["avg_tokens_per_run"] is None

    def test_stats_tokens_by_model_key_present(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """The stats response includes a tokens_by_model list."""
        resp = client.get("/api/runs/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens_by_model" in data
        assert isinstance(data["tokens_by_model"], list)

    def test_stats_tokens_by_model_populated(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """tokens_by_model is populated when task_results include token_usage."""
        task_results = {
            "t1": {
                "status": "success",
                "command": "claude --model sonnet",
                "token_usage": {"total_tokens": 3000},
            }
        }
        _write_manifest(
            tmp_path, "run-a", _make_manifest(task_results=task_results)
        )

        resp = client.get("/api/runs/stats")
        data = resp.json()
        tokens_by_model = data["tokens_by_model"]
        assert len(tokens_by_model) >= 1
        entry = tokens_by_model[0]
        assert "model" in entry
        assert "total_tokens" in entry
        assert "task_count" in entry
        assert "avg_tokens" in entry

    def test_stats_tokens_by_model_avg_calculated(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """tokens_by_model.avg_tokens = total_tokens / task_count."""
        task_results = {
            "t1": {
                "status": "success",
                "command": "claude --model haiku",
                "token_usage": {"total_tokens": 4000},
            },
            "t2": {
                "status": "success",
                "command": "claude --model haiku",
                "token_usage": {"total_tokens": 2000},
            },
        }
        _write_manifest(
            tmp_path, "run-a", _make_manifest(task_results=task_results)
        )

        resp = client.get("/api/runs/stats")
        data = resp.json()
        by_model = data["tokens_by_model"]
        assert len(by_model) >= 1
        entry = next(
            (e for e in by_model if e["task_count"] == 2), None
        )
        assert entry is not None
        assert entry["total_tokens"] == 6000
        assert entry["avg_tokens"] == 3000

    def test_recent_runs_include_total_tokens(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """recent_runs entries carry total_tokens from the manifest."""
        _write_manifest(tmp_path, "run-a", _make_manifest(total_tokens=777))

        resp = client.get("/api/runs/stats")
        data = resp.json()
        recent = data["recent_runs"]
        assert len(recent) == 1
        assert recent[0]["total_tokens"] == 777

    def test_stats_total_tokens_fallback_from_task_results(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """total_tokens falls back to summing task token_usage when manifest lacks it."""
        task_results = {
            "t1": {
                "status": "success",
                "command": "echo hi",
                "token_usage": {"total_tokens": 400},
            },
            "t2": {
                "status": "success",
                "command": "echo world",
                "token_usage": {"total_tokens": 600},
            },
        }
        # Manifest without top-level total_tokens
        manifest = _make_manifest(task_results=task_results)
        _write_manifest(tmp_path, "run-a", manifest)

        resp = client.get("/api/runs/stats")
        data = resp.json()
        # _total_tokens_from_manifest falls back to summing task token_usage values
        assert data["total_tokens"] == 1000
