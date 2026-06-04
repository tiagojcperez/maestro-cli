"""Coverage tests for pure helper functions in web/routes_api.py.

These target the error/edge branches of the manifest-parsing and
activity-feed helpers that the happy-path web tests never exercise:
duplicate run-root dedup, malformed manifest values (non-dict task
results, un-castable cost/token totals), CLI-token quote stripping,
codex inference fallback, task-graph normalization skips, the
engine:model runtime formatting, every activity-message event type,
and the activity-feed builder including its >limit truncation.

All boundaries are mocked: no engine CLI, no network, no real git.
The only filesystem touched is ``tmp_path``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.models import EventRecord
from maestro_cli.web import routes_api


def _event(
    event_type: str,
    payload: dict[str, object] | None = None,
    *,
    sequence: int = 0,
    timestamp: str = "2026-06-04T00:00:00+00:00",
) -> EventRecord:
    """Build an EventRecord for activity-feed tests."""
    return EventRecord(
        sequence=sequence,
        event_type=event_type,
        timestamp=timestamp,
        payload=payload or {},
        prev_hash="",
        event_hash="",
    )


# ---------------------------------------------------------------------------
# _discover_run_roots — duplicate candidate is skipped (dedup `continue`)
# ---------------------------------------------------------------------------

class TestDiscoverRunRoots:
    def test_duplicate_candidates_deduplicated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base_a = Path("/proj/a")
        base_b = Path("/proj/b")
        dup = Path("/runs/shared")
        unique = Path("/runs/only-b")

        monkeypatch.setattr(
            routes_api, "get_project_roots", lambda: [base_a, base_b]
        )

        def _fake_discover(base: Path) -> list[Path]:
            if base == base_a:
                return [dup]
            return [dup, unique]  # dup repeats -> hits the `continue`

        monkeypatch.setattr(routes_api, "discover_run_roots", _fake_discover)

        roots = routes_api._discover_run_roots()
        # Each distinct path appears exactly once despite the repeat.
        assert roots == [dup, unique]


# ---------------------------------------------------------------------------
# _is_dry_run_from_manifest — non-dict task result is skipped
# ---------------------------------------------------------------------------

class TestIsDryRunFromManifest:
    def test_non_dict_task_result_is_skipped(self) -> None:
        manifest = {
            "task_results": {
                "t1": "not-a-dict",  # skipped via `continue`
                "t2": {"status": "dry_run"},
            }
        }
        assert routes_api._is_dry_run_from_manifest(manifest) is True

    def test_only_non_dict_results_means_no_statuses(self) -> None:
        manifest = {"task_results": {"t1": ["also", "not", "a", "dict"]}}
        # No string statuses collected -> falls to the "no statuses" return.
        assert routes_api._is_dry_run_from_manifest(manifest) is False


# ---------------------------------------------------------------------------
# _duration_from_manifest — invalid isoformat hits except -> None
# ---------------------------------------------------------------------------

class TestDurationFromManifest:
    def test_invalid_iso_returns_none(self) -> None:
        manifest = {
            "started_at": "not-a-timestamp",
            "finished_at": "also-bad",
        }
        assert routes_api._duration_from_manifest(manifest) is None

    def test_valid_iso_returns_positive_duration(self) -> None:
        manifest = {
            "started_at": "2026-06-04T00:00:00+00:00",
            "finished_at": "2026-06-04T00:00:05+00:00",
        }
        assert routes_api._duration_from_manifest(manifest) == 5.0


# ---------------------------------------------------------------------------
# _total_cost_from_manifest
# ---------------------------------------------------------------------------

class TestTotalCostFromManifest:
    def test_total_uncastable_falls_through_to_task_results(self) -> None:
        # total is non-None but not float-able -> except: pass, then sums tasks.
        manifest = {
            "total_cost_usd": object(),
            "task_results": {"t1": {"cost_usd": 1.5}, "t2": {"cost_usd": 0.5}},
        }
        assert routes_api._total_cost_from_manifest(manifest) == 2.0

    def test_task_results_not_dict_returns_none(self) -> None:
        manifest = {"task_results": ["not", "a", "dict"]}
        assert routes_api._total_cost_from_manifest(manifest) is None

    def test_non_dict_task_result_and_bad_cost_are_skipped(self) -> None:
        manifest = {
            "task_results": {
                "bad": "string-not-dict",      # malformed entry -> skipped
                "uncastable": {"cost_usd": object()},  # cost cast fails -> continue
                "good": {"cost_usd": 0.25},
            }
        }
        assert routes_api._total_cost_from_manifest(manifest) == 0.25

    def test_no_costs_returns_none(self) -> None:
        manifest = {"task_results": {"t1": {"cost_usd": None}}}
        assert routes_api._total_cost_from_manifest(manifest) is None


# ---------------------------------------------------------------------------
# _total_tokens_from_manifest
# ---------------------------------------------------------------------------

class TestTotalTokensFromManifest:
    def test_total_uncastable_falls_through_to_task_results(self) -> None:
        manifest = {
            "total_tokens": object(),  # int(object()) -> TypeError -> pass
            "task_results": {
                "t1": {"token_usage": {"total_tokens": 100}},
                "t2": {"token_usage": {"total_tokens": 50}},
            },
        }
        assert routes_api._total_tokens_from_manifest(manifest) == 150

    def test_task_results_not_dict_returns_none(self) -> None:
        manifest = {"total_tokens": None, "task_results": ["x"]}
        assert routes_api._total_tokens_from_manifest(manifest) is None

    def test_non_dict_result_none_token_and_uncastable_skipped(self) -> None:
        manifest = {
            "task_results": {
                "bad": "not-a-dict",                       # malformed entry -> skipped
                "none_tok": {"token_usage": {"total_tokens": None}},  # malformed entry -> skipped
                "uncastable": {"token_usage": {"total_tokens": object()}},  # cast fails
                "good": {"token_usage": {"total_tokens": 42}},
            }
        }
        assert routes_api._total_tokens_from_manifest(manifest) == 42

    def test_no_tokens_returns_none(self) -> None:
        manifest = {"task_results": {"t1": {"token_usage": {}}}}
        assert routes_api._total_tokens_from_manifest(manifest) is None


# ---------------------------------------------------------------------------
# _clean_cli_token — strips matching surrounding quotes
# ---------------------------------------------------------------------------

class TestCleanCliToken:
    def test_strips_double_quotes(self) -> None:
        assert routes_api._clean_cli_token('"opus"') == "opus"

    def test_strips_single_quotes(self) -> None:
        assert routes_api._clean_cli_token("'sonnet'") == "sonnet"

    def test_unquoted_token_unchanged(self) -> None:
        assert routes_api._clean_cli_token("haiku") == "haiku"


# ---------------------------------------------------------------------------
# _infer_model_from_task_result — codex / claude name fallbacks
# ---------------------------------------------------------------------------

class TestInferModel:
    def test_quoted_model_flag_for_claude(self) -> None:
        result = {"command": 'claude --model "opus" --print'}
        assert routes_api._infer_model_from_task_result(result) == "opus"

    def test_codex_dash_m_flag(self) -> None:
        result = {"command": "codex exec -m gpt-5.4-codex prompt"}
        assert routes_api._infer_model_from_task_result(result) == "gpt-5.4-codex"

    def test_codex_name_fallback(self) -> None:
        # No model flag, but the word "codex" appears -> returns "codex".
        result = {"command": "run codex with a plan"}
        assert routes_api._infer_model_from_task_result(result) == "codex"

    def test_claude_name_fallback(self) -> None:
        result = {"command": "invoke claude here"}
        assert routes_api._infer_model_from_task_result(result) == "claude"

    def test_no_command_returns_none(self) -> None:
        assert routes_api._infer_model_from_task_result({"command": ""}) is None

    def test_unrecognised_command_returns_none(self) -> None:
        assert routes_api._infer_model_from_task_result({"command": "echo hi"}) is None


# ---------------------------------------------------------------------------
# _build_task_graph_from_tasks — non-string id is skipped
# ---------------------------------------------------------------------------

class _FakeTask:
    def __init__(self, task_id: object, **attrs: object) -> None:
        self.id = task_id
        for key, value in attrs.items():
            setattr(self, key, value)


class TestBuildTaskGraphFromTasks:
    def test_non_string_id_skipped(self) -> None:
        tasks = [
            _FakeTask(None),       # skipped via `continue`
            _FakeTask(""),         # empty string also skipped
            _FakeTask("real", description="do", depends_on=["x", 7], engine="claude"),
        ]
        graph = routes_api._build_task_graph_from_tasks(tasks)
        assert set(graph.keys()) == {"real"}
        entry = graph["real"]
        assert entry["description"] == "do"
        # Non-string deps are filtered out.
        assert entry["depends_on"] == ["x"]
        assert entry["engine"] == "claude"


# ---------------------------------------------------------------------------
# _normalize_task_graph — non-string id is skipped
# ---------------------------------------------------------------------------

class TestNormalizeTaskGraph:
    def test_non_dict_input_returns_empty(self) -> None:
        assert routes_api._normalize_task_graph(["not", "a", "dict"]) == {}

    def test_non_string_id_skipped(self) -> None:
        raw = {
            "": {"description": "skip me"},  # empty id skipped
            123: {"description": "non-str id"},  # non-string id skipped
            "keep": {
                "description": "kept",
                "depends_on": ["a", None],
                "agent": "security-engineer",
                "engine": "claude",
                "model": "opus",
                "allow_failure": True,
                "worktree": True,
            },
        }
        normalized = routes_api._normalize_task_graph(raw)
        assert set(normalized.keys()) == {"keep"}
        kept = normalized["keep"]
        assert kept["depends_on"] == ["a"]
        assert kept["agent"] == "security-engineer"
        assert kept["allow_failure"] is True
        assert kept["worktree"] is True

    def test_non_dict_meta_defaults_applied(self) -> None:
        normalized = routes_api._normalize_task_graph({"t1": "meta-not-a-dict"})
        assert normalized["t1"]["description"] == ""
        assert normalized["t1"]["depends_on"] == []
        assert normalized["t1"]["agent"] is None


# ---------------------------------------------------------------------------
# _task_runtime — engine:model formatting and inference fallback
# ---------------------------------------------------------------------------

class TestTaskRuntime:
    def test_engine_and_model_combined(self) -> None:
        meta = {"engine": "claude", "model": "sonnet"}
        assert routes_api._task_runtime(meta, None) == "claude:sonnet"

    def test_engine_only_returns_engine(self) -> None:
        meta = {"engine": "codex", "model": ""}
        assert routes_api._task_runtime(meta, None) == "codex"

    def test_no_engine_infers_from_result(self) -> None:
        meta: dict[str, object] = {}
        result = {"command": 'claude --model "haiku"'}
        assert routes_api._task_runtime(meta, result) == "haiku"

    def test_no_engine_no_inference_returns_none(self) -> None:
        assert routes_api._task_runtime({}, {"command": "echo hi"}) is None

    def test_no_engine_no_result_returns_none(self) -> None:
        assert routes_api._task_runtime({}, None) is None


# ---------------------------------------------------------------------------
# _format_activity_message — every event-type branch
# ---------------------------------------------------------------------------

class TestFormatActivityMessage:
    def test_task_start(self) -> None:
        msg = routes_api._format_activity_message("task_start", {"task_id": "t1"})
        assert msg == "t1 started"

    def test_task_complete_with_status(self) -> None:
        msg = routes_api._format_activity_message(
            "task_complete", {"task_id": "t1", "status": "success"}
        )
        assert "finished as success" in msg

    def test_task_complete_without_status(self) -> None:
        # Missing task_id -> falls back to label "task".
        msg = routes_api._format_activity_message("task_complete", {})
        assert msg == "task finished"

    def test_task_skip_with_reason(self) -> None:
        msg = routes_api._format_activity_message(
            "task_skip", {"task_id": "t1", "reason": "when=false"}
        )
        assert "skipped: when=false" in msg

    def test_task_skip_without_reason(self) -> None:
        msg = routes_api._format_activity_message("task_skip", {"task_id": "t1"})
        assert msg == "t1 skipped"

    def test_task_progress_with_pct_and_step(self) -> None:
        msg = routes_api._format_activity_message(
            "task_progress", {"task_id": "t1", "pct": 42.7, "step": "compiling"}
        )
        assert "progress 42%" in msg
        assert "(compiling)" in msg

    def test_task_progress_without_pct(self) -> None:
        msg = routes_api._format_activity_message(
            "task_progress", {"task_id": "t1", "pct": "bad"}
        )
        assert msg == "t1 progress update"

    def test_approval_required(self) -> None:
        msg = routes_api._format_activity_message(
            "approval_required", {"task_id": "t1"}
        )
        assert "waiting for approval" in msg

    def test_approval_response_approved(self) -> None:
        msg = routes_api._format_activity_message(
            "approval_response", {"task_id": "t1", "approved": True}
        )
        assert msg == "t1 approved"

    def test_approval_response_denied(self) -> None:
        msg = routes_api._format_activity_message(
            "approval_response", {"task_id": "t1", "approved": False}
        )
        assert msg == "t1 denied"

    def test_approval_response_other(self) -> None:
        msg = routes_api._format_activity_message(
            "approval_response", {"task_id": "t1", "approved": None}
        )
        assert "approval updated" in msg

    def test_budget_exceeded(self) -> None:
        msg = routes_api._format_activity_message("budget_exceeded", {})
        assert msg == "Budget exceeded"

    def test_policy_violation_with_action(self) -> None:
        msg = routes_api._format_activity_message(
            "policy_violation", {"action": "block"}
        )
        assert "Policy violation (block)" in msg

    def test_policy_violation_without_action(self) -> None:
        msg = routes_api._format_activity_message("policy_violation", {})
        assert "Policy violation detected" in msg

    def test_unknown_event_returns_none(self) -> None:
        assert routes_api._format_activity_message("run_start", {}) is None


# ---------------------------------------------------------------------------
# _load_activity_feed
# ---------------------------------------------------------------------------

class TestLoadActivityFeed:
    def test_missing_events_file_returns_empty(self, tmp_path: Path) -> None:
        assert routes_api._load_activity_feed(tmp_path, {}) == []

    def test_replay_oserror_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "events.jsonl").write_text("{}\n", encoding="utf-8")

        def _boom(_path: Path) -> list[EventRecord]:
            raise OSError("disk gone")

        monkeypatch.setattr(routes_api, "replay_events", _boom)
        assert routes_api._load_activity_feed(tmp_path, {}) == []

    def test_builds_entries_with_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "events.jsonl").write_text("ignored\n", encoding="utf-8")
        records = [
            _event("task_start", {"task_id": "t1"}),
            _event("run_start", {}),  # no message -> filtered out via `continue`
        ]
        monkeypatch.setattr(routes_api, "replay_events", lambda _p: records)

        task_graph = {"t1": {"agent": "qa-engineer"}}
        feed = routes_api._load_activity_feed(tmp_path, task_graph)
        assert len(feed) == 1
        entry = feed[0]
        assert entry["task_id"] == "t1"
        assert entry["owner"] == "qa-engineer"
        assert entry["event"] == "task_start"
        assert "started" in entry["message"]

    def test_non_string_task_id_yields_none_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "events.jsonl").write_text("x\n", encoding="utf-8")
        # budget_exceeded produces a message but its payload has no string task_id.
        records = [_event("budget_exceeded", {})]
        monkeypatch.setattr(routes_api, "replay_events", lambda _p: records)

        feed = routes_api._load_activity_feed(tmp_path, {})
        assert len(feed) == 1
        assert feed[0]["task_id"] is None
        assert feed[0]["owner"] is None

    def test_feed_truncated_to_activity_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "events.jsonl").write_text("x\n", encoding="utf-8")
        limit = routes_api._ACTIVITY_LIMIT
        records = [
            _event("task_start", {"task_id": f"t{i}"}, sequence=i)
            for i in range(limit + 5)
        ]
        monkeypatch.setattr(routes_api, "replay_events", lambda _p: records)

        feed = routes_api._load_activity_feed(tmp_path, {})
        assert len(feed) == limit
        # Tail kept: last task id should be the final record's id.
        assert feed[-1]["task_id"] == f"t{limit + 4}"
