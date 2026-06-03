from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.cli import _build_parser, main
from maestro_cli.loader import load_plan
from maestro_cli.models import PlanDefaults, PlanSpec, TaskResult, TaskSpec
from maestro_cli.scheduler import _post_completion_webhook, run_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PLAN_WITH_WEBHOOK_YAML = """\
version: 1
name: webhook-plan
webhook_url: "https://example.com/from-plan"
tasks:
  - id: t1
    command: "echo hello"
"""

_PLAN_WITHOUT_WEBHOOK_YAML = """\
version: 1
name: no-webhook-plan
tasks:
  - id: t1
    command: "echo hello"
"""


def _write_plan(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _make_task(task_id: str, command: str = "echo ok") -> TaskSpec:
    return TaskSpec(id=task_id, description=f"task {task_id}", command=command)


def _make_plan(
    tasks: list[TaskSpec],
    name: str = "test-plan",
    webhook_url: str | None = None,
    source_path: Path | None = None,
) -> PlanSpec:
    return PlanSpec(
        version=1,
        name=name,
        defaults=PlanDefaults(),
        tasks=tasks,
        webhook_url=webhook_url,
        source_path=source_path,
    )


def _mock_success_execute(
    plan: Any,
    task: Any,
    run_path: Path,
    dry_run: bool = False,
    execution_profile: str = "plan",
    upstream_results: Any = None,
    context_synthesis: str = "",
    workspace_brief: str = "",
    **kwargs,
) -> TaskResult:
    now = datetime.now(UTC)
    status = "dry_run" if dry_run else "success"
    result = TaskResult(
        task_id=task.id,
        status=status,
        exit_code=0,
        started_at=now,
        finished_at=now,
        duration_sec=0.01,
        command=f"echo {task.id}",
        log_path=run_path / f"{task.id}.log",
        result_path=run_path / f"{task.id}.result.json",
        message="ok",
    )
    result.log_path.write_text(f"status={status}\n", encoding="utf-8")
    result.result_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    return result


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


class TestWebhookCliParsing:
    def test_webhook_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            ["run", "plan.yaml", "--webhook", "https://hook.example.com/notify"]
        )
        assert args.webhook == "https://hook.example.com/notify"

    def test_webhook_flag_default_is_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.webhook is None


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


class TestWebhookYamlParsing:
    def test_webhook_url_parsed_from_yaml(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _PLAN_WITH_WEBHOOK_YAML)
        plan = load_plan(plan_file)
        assert plan.webhook_url == "https://example.com/from-plan"

    def test_webhook_url_none_when_not_set(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _PLAN_WITHOUT_WEBHOOK_YAML)
        plan = load_plan(plan_file)
        assert plan.webhook_url is None


# ---------------------------------------------------------------------------
# _post_completion_webhook function
# ---------------------------------------------------------------------------


class TestPostCompletionWebhook:
    def test_sends_post_with_json_content_type(self) -> None:
        """Webhook POST sets Content-Type: application/json."""
        captured_requests: list[urllib.request.Request] = []

        def mock_urlopen(req: urllib.request.Request, timeout: int) -> MagicMock:
            captured_requests.append(req)
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            _post_completion_webhook("https://example.com/hook", {"key": "value"})

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert req.get_header("Content-type") == "application/json"
        assert req.get_method() == "POST"

    def test_sends_correct_payload(self) -> None:
        """Webhook POST body is JSON-encoded payload."""
        captured_bodies: list[bytes] = []

        def mock_urlopen(req: urllib.request.Request, timeout: int) -> MagicMock:
            captured_bodies.append(req.data)
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        payload = {"plan_name": "my-plan", "success": True, "ok_count": 3}
        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            _post_completion_webhook("https://example.com/hook", payload)

        assert len(captured_bodies) == 1
        sent = json.loads(captured_bodies[0].decode("utf-8"))
        assert sent["plan_name"] == "my-plan"
        assert sent["success"] is True

    def test_returns_http_status_code(self) -> None:
        """Return value is the HTTP status code from the response."""
        def mock_urlopen(req: Any, timeout: int) -> MagicMock:
            resp = MagicMock()
            resp.status = 201
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            code = _post_completion_webhook("https://example.com/hook", {})

        assert code == 201


# ---------------------------------------------------------------------------
# Webhook integration with run_plan
# ---------------------------------------------------------------------------


class TestWebhookInRunPlan:
    def test_webhook_not_called_when_url_is_none(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """No HTTP request is made when webhook_url is None."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)

        called = []

        def spy_urlopen(req: Any, timeout: int) -> Any:
            called.append(req)
            raise AssertionError("urlopen should not be called")

        with patch("urllib.request.urlopen", side_effect=spy_urlopen):
            run_plan(
                _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml"),
                run_dir_override=str(tmp_path / "runs"),
                webhook_url=None,
            )

        assert called == []

    def test_webhook_called_with_run_data(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """Webhook is POSTed with correct plan/run fields."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)

        payloads: list[dict[str, Any]] = []

        def mock_urlopen(req: urllib.request.Request, timeout: int) -> MagicMock:
            payloads.append(json.loads(req.data.decode("utf-8")))
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            run_plan(
                _make_plan(
                    [_make_task("t1")],
                    name="hook-plan",
                    source_path=tmp_path / "plan.yaml",
                ),
                run_dir_override=str(tmp_path / "runs"),
                webhook_url="https://example.com/notify",
            )

        assert len(payloads) == 1
        p = payloads[0]
        assert p["plan_name"] == "hook-plan"
        assert p["success"] is True
        assert "run_id" in p
        assert "ok_count" in p
        assert "failed_count" in p
        assert "skipped_count" in p
        assert "duration_sec" in p

    def test_webhook_failure_does_not_fail_run(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """A failing webhook POST does not cause the run result to be unsuccessful."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)

        def mock_urlopen(req: Any, timeout: int) -> None:
            raise urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = run_plan(
                _make_plan(
                    [_make_task("t1")],
                    source_path=tmp_path / "plan.yaml",
                ),
                run_dir_override=str(tmp_path / "runs"),
                webhook_url="https://example.com/notify",
            )

        assert result.success is True

    def test_cli_webhook_overrides_plan_webhook(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """--webhook CLI flag takes precedence over plan webhook_url."""
        plan_file = _write_plan(tmp_path, _PLAN_WITH_WEBHOOK_YAML)
        called_urls: list[str] = []

        def mock_urlopen(req: urllib.request.Request, timeout: int) -> MagicMock:
            called_urls.append(req.full_url)
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            main([
                "run", str(plan_file),
                "--dry-run",
                "--run-dir", str(tmp_path / "runs"),
                "--webhook", "https://override.example.com/hook",
            ])

        assert len(called_urls) == 1
        assert called_urls[0] == "https://override.example.com/hook"

    def test_webhook_event_emitted_on_success(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        """A webhook event is written to events.jsonl when delivery succeeds."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)

        def mock_urlopen(req: Any, timeout: int) -> MagicMock:
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = run_plan(
                _make_plan(
                    [_make_task("t1")],
                    source_path=tmp_path / "plan.yaml",
                ),
                run_dir_override=str(tmp_path / "runs"),
                webhook_url="https://example.com/notify",
            )

        events_text = (result.run_path / "events.jsonl").read_text(encoding="utf-8")
        events = _parse_jsonl(events_text)
        webhook_events = [e for e in events if e["event"] == "webhook"]
        assert len(webhook_events) == 1
        assert webhook_events[0]["status"] == "delivered"
        assert webhook_events[0]["url"] == "https://example.com/notify"
        assert "http_status" in webhook_events[0]

    def test_webhook_event_emitted_on_failure(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """A webhook event with status=failed is written when delivery fails."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)

        def mock_urlopen(req: Any, timeout: int) -> None:
            raise urllib.error.URLError("timeout")

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            result = run_plan(
                _make_plan(
                    [_make_task("t1")],
                    source_path=tmp_path / "plan.yaml",
                ),
                run_dir_override=str(tmp_path / "runs"),
                webhook_url="https://example.com/notify",
            )

        events_text = (result.run_path / "events.jsonl").read_text(encoding="utf-8")
        events = _parse_jsonl(events_text)
        webhook_events = [e for e in events if e["event"] == "webhook"]
        assert len(webhook_events) == 1
        assert webhook_events[0]["status"] == "failed"
        assert "error" in webhook_events[0]
