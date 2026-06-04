from __future__ import annotations

"""Targeted line-coverage tests for ``maestro_cli.runners``.

Each test drives a specific previously-uncovered branch in the firewall,
context-shaping, retry-compression, phantom-workspace, and population-search
helpers. All external boundaries (subprocess, ``execute_task``) are mocked.
"""

from pathlib import Path

import pytest

import maestro_cli.runners as runners
from maestro_cli.models import (
    PlanDefaults,
    PlanSpec,
    PopulationSpec,
    TaskResult,
    TaskSpec,
)
from maestro_cli.runners import (
    _apply_mcp_description_firewall,
    _apply_untrusted_content_firewall,
    _build_layered_context,
    _build_selective_context,
    _commit_phantom_workspace,
    _compress_context_for_retry,
    _extract_l1_sections,
    _prune_low_signal_sections,
    _resolve_task_mcp_servers,
    _run_firewall_pass2,
    _run_population_search,
)


def _make_plan(firewall_model: str | None = None) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="cov-plan",
        defaults=PlanDefaults(),
        tasks=[],
        firewall_model=firewall_model,
    )


# ---------------------------------------------------------------------------
# _run_firewall_pass2 — fail-fast guard + verdict normalization
# ---------------------------------------------------------------------------


class TestRunFirewallPass2:
    def test_empty_model_returns_default_allow(self) -> None:
        """No model -> immediate default ``allow`` decision (early guard)."""
        decision = _run_firewall_pass2("", "src", "some content")
        assert decision.verdict == "allow"
        assert decision.category == ""

    def test_blank_content_returns_default_allow(self) -> None:
        """Whitespace-only content -> immediate default decision."""
        decision = _run_firewall_pass2("haiku", "src", "   \n  ")
        assert decision.verdict == "allow"

    def test_invalid_verdict_normalized_to_allow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A classifier verdict outside the allowed set falls back to ``allow``."""

        class _FakeProc:
            stdout = '{"verdict": "explode", "category": "weird", "reason": "r"}'

        def fake_run(*args: object, **kwargs: object) -> _FakeProc:
            return _FakeProc()

        monkeypatch.setattr(runners.subprocess, "run", fake_run)
        decision = _run_firewall_pass2("haiku", "src", "real content here")
        assert decision.verdict == "allow"
        # category/reason still propagated from the (otherwise valid) payload
        assert decision.category == "weird"
        assert decision.reason == "r"


# ---------------------------------------------------------------------------
# _apply_mcp_description_firewall — rewrite branch with empty sanitized
# ---------------------------------------------------------------------------


class TestApplyMcpDescriptionFirewall:
    def test_rewrite_with_empty_sanitized_withholds_description(self) -> None:
        """Content that sanitizes to empty but yields findings -> ``rewrite``
        decision with ``not sanitized`` -> description withheld message."""
        plan = _make_plan()  # no firewall_model -> heuristic decision from findings
        content = "mcp__foo__bar"  # -> sanitized="", findings=["mcp_tool_handle"]
        text, withheld, verdict = _apply_mcp_description_firewall(
            plan, content, server_name="srv"
        )
        assert withheld is True
        assert verdict == "rewrite"
        assert "withheld by semantic firewall" in text

    def test_rewrite_with_nonempty_sanitized_returns_sanitized(self) -> None:
        """A finding that leaves residual text returns the sanitized text."""
        plan = _make_plan()
        # 'javascript:do' triggers dangerous_scheme but leaves residual text.
        content = "click javascript:run now to continue please"
        text, withheld, verdict = _apply_mcp_description_firewall(
            plan, content, server_name="srv"
        )
        assert withheld is True
        assert verdict == "rewrite"
        assert "withheld by semantic firewall" not in text

    def test_block_verdict_withholds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pass-2 ``block`` decision withholds the description."""
        plan = _make_plan(firewall_model="haiku")
        monkeypatch.setattr(
            runners,
            "_run_firewall_pass2",
            lambda *a, **k: runners._FirewallDecision(
                verdict="block", category="bad"
            ),
        )
        text, withheld, verdict = _apply_mcp_description_firewall(
            plan, "harmless description text", server_name="srv"
        )
        assert withheld is True
        assert verdict == "block"
        assert "withheld by semantic firewall: bad" in text


# ---------------------------------------------------------------------------
# _apply_untrusted_content_firewall — block path (reachable companion branch)
# ---------------------------------------------------------------------------


class TestApplyUntrustedContentFirewall:
    def test_block_verdict_returns_blocked_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = _make_plan(firewall_model="haiku")
        monkeypatch.setattr(
            runners,
            "_run_firewall_pass2",
            lambda *a, **k: runners._FirewallDecision(
                verdict="block", category="evil"
            ),
        )
        out = _apply_untrusted_content_firewall(
            plan, TaskSpec(id="t"), "ordinary upstream text", source_label="up"
        )
        assert "blocked up: evil" in out

    def test_rewrite_with_residual_returns_sanitized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = _make_plan(firewall_model="haiku")
        monkeypatch.setattr(
            runners,
            "_run_firewall_pass2",
            lambda *a, **k: runners._FirewallDecision(verdict="rewrite"),
        )
        out = _apply_untrusted_content_firewall(
            plan, TaskSpec(id="t"), "ordinary upstream text", source_label="up"
        )
        # sanitized is non-empty so the rewrite-with-empty branch is skipped.
        assert "ordinary upstream text" in out


# ---------------------------------------------------------------------------
# _resolve_task_mcp_servers — missing server skip + role-denied raise
# ---------------------------------------------------------------------------


class TestResolveTaskMcpServers:
    def test_unknown_server_reference_is_skipped(self) -> None:
        """A ``mcp_tools`` name with no matching server definition is skipped."""
        from maestro_cli.models import MCPServerSpec

        server = MCPServerSpec(name="known", transport="stdio", command=["x"])
        plan = _make_plan()
        plan.mcp_servers = [server]
        task = TaskSpec(id="t", engine="claude", mcp_tools=["missing", "known"])
        resolved = _resolve_task_mcp_servers(plan, task)
        assert [s.name for s in resolved] == ["known"]

    def test_role_not_allowed_raises(self) -> None:
        """A task whose agent role is outside ``allowed_task_roles`` raises."""
        from maestro_cli.errors import TaskExecutionError
        from maestro_cli.models import MCPServerSpec

        server = MCPServerSpec(
            name="srv",
            transport="stdio",
            command=["x"],
            allowed_task_roles=["security-engineer"],
        )
        plan = _make_plan()
        plan.mcp_servers = [server]
        task = TaskSpec(id="t", engine="claude", mcp_tools=["srv"], agent="qa")
        with pytest.raises(TaskExecutionError, match="cannot use MCP server"):
            _resolve_task_mcp_servers(plan, task)


# ---------------------------------------------------------------------------
# _extract_l1_sections — break when a high-signal line overflows the budget
# ---------------------------------------------------------------------------


class TestExtractL1Sections:
    def test_high_signal_line_overflow_breaks(self) -> None:
        """A captured heading consumes budget, then a long bullet line
        overflows and triggers the high-signal ``break``."""
        text = "# Heading\n" + "- " + ("x" * 400) + "\n"
        out = _extract_l1_sections(text, max_chars=40)
        assert "# Heading" in out
        # The oversized bullet was not appended.
        assert "xxxx" not in out


# ---------------------------------------------------------------------------
# _build_selective_context — skip upstreams with whitespace-only text
# ---------------------------------------------------------------------------


class TestBuildSelectiveContext:
    def test_empty_upstream_text_is_skipped(self) -> None:
        """An upstream whose text is whitespace-only is skipped (no chunk)."""
        out = _build_selective_context(
            {
                "blankid": "   \n\t  ",
                "richid": "alpha beta gamma keyword keyword keyword\n" * 5,
            },
            budget_tokens=200,
            intent_keywords={"keyword"},
        )
        # The blank upstream's whitespace body must not surface.
        assert "blankid" not in out
        assert "keyword" in out


# ---------------------------------------------------------------------------
# _prune_low_signal_sections — scoring branches (blank / numbered / keyword / fence)
# ---------------------------------------------------------------------------


class TestPruneLowSignalSections:
    def test_all_scoring_branches_execute(self) -> None:
        """Craft input lines that exercise each scoring branch."""
        text = "\n".join(
            [
                "ordinary descriptive prose line content",  # else -> 1.0
                "",  # blank -> -1.0
                "1. first numbered list item content",  # numbered -> 6.0
                "1) second numbered variant content",  # numbered -> 6.0
                "this sentence mentions an error inside it",  # keyword -> 5.0
                "```",  # fence -> 3.0
                "=== boundary marker text ===",  # fence/divider -> 3.0
            ]
        )
        out = _prune_low_signal_sections(text, target_chars=10000)
        # With a generous budget every line is retained.
        assert "first numbered list item" in out
        assert "mentions an error" in out
        assert "boundary marker" in out

    def test_compaction_marker_added_when_lines_dropped(self) -> None:
        """A tight budget drops low-signal lines and appends the marker."""
        lines = ["# keep me"] + [f"low signal filler line {i}" for i in range(50)]
        out = _prune_low_signal_sections("\n".join(lines), target_chars=30)
        assert "# keep me" in out
        assert "chars removed" in out


# ---------------------------------------------------------------------------
# _compress_context_for_retry — short-target + small-tail branches
# ---------------------------------------------------------------------------


class TestCompressContextForRetry:
    def test_short_target_returns_tail_slice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the computed target is tiny, return a plain tail slice."""
        # Lower the floor so target_len can drop below marker-len + 64.
        monkeypatch.setattr(runners, "_CONTEXT_RETRY_MIN_CHARS", 70)
        text = "abcdefg" * 200
        out = _compress_context_for_retry(text, compression_level=8)
        # Tail slice: no compression marker is inserted.
        assert runners._CONTEXT_RETRY_MARKER not in out
        assert out == text[-len(out):]

    def test_small_tail_clamped_to_minimum(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When tail_len would fall below 64, it is clamped and head shrinks."""
        monkeypatch.setattr(runners, "_CONTEXT_RETRY_MIN_CHARS", 108)
        text = "z" * 4000
        out = _compress_context_for_retry(text, compression_level=8)
        assert runners._CONTEXT_RETRY_MARKER in out
        head, _, tail = out.partition(runners._CONTEXT_RETRY_MARKER)
        assert len(tail) == 64


# ---------------------------------------------------------------------------
# _commit_phantom_workspace — early return when phantom dir is absent
# ---------------------------------------------------------------------------


class TestCommitPhantomWorkspace:
    def test_missing_phantom_dir_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        target = tmp_path / "target"
        target.mkdir()
        assert _commit_phantom_workspace(missing, target) == []

    def test_files_are_copied_to_target(self, tmp_path: Path) -> None:
        phantom = tmp_path / "phantom"
        (phantom / "sub").mkdir(parents=True)
        (phantom / "a.txt").write_text("A", encoding="utf-8")
        (phantom / "sub" / "b.txt").write_text("B", encoding="utf-8")
        target = tmp_path / "target"
        target.mkdir()
        committed = _commit_phantom_workspace(phantom, target)
        assert (target / "a.txt").read_text(encoding="utf-8") == "A"
        assert (target / "sub" / "b.txt").read_text(encoding="utf-8") == "B"
        assert len(committed) == 2


# ---------------------------------------------------------------------------
# _run_population_search — parallel path + first_passing/majority fallbacks
# ---------------------------------------------------------------------------


def _pop_task(strategy: str, parallel: bool) -> TaskSpec:
    return TaskSpec(
        id="poptask",
        engine="claude",
        prompt="do it",
        population=PopulationSpec(
            candidates=["haiku", "sonnet"],
            strategy=strategy,  # type: ignore[arg-type]
            parallel=parallel,
        ),
    )


class TestRunPopulationSearch:
    def test_parallel_first_passing_selects_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Parallel execution path: one candidate raises (swallowed by the
        ``except`` in the futures loop) while the other succeeds and wins."""

        def fake_execute(plan: PlanSpec, task: TaskSpec, run_path: Path, **kw: object) -> TaskResult:
            if task.model == "haiku":
                raise RuntimeError("boom in candidate")
            return TaskResult(task_id=task.id, status="success")

        monkeypatch.setattr(runners, "execute_task", fake_execute)
        events: list[tuple[str, dict[str, object]]] = []
        result = _run_population_search(
            _make_plan(),
            _pop_task("first_passing", parallel=True),
            tmp_path,
            "plan",
            None,
            "",
            "",
            lambda name, payload: events.append((name, payload)),
            None,
            None,
        )
        assert result.status == "success"
        assert any(name == "population_selected" for name, _ in events)

    def test_first_passing_no_success_returns_first_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No passing candidate -> first_passing returns the first result."""

        def fake_execute(plan: PlanSpec, task: TaskSpec, run_path: Path, **kw: object) -> TaskResult:
            return TaskResult(task_id=task.id, status="failed", message=task.model)

        monkeypatch.setattr(runners, "execute_task", fake_execute)
        result = _run_population_search(
            _make_plan(),
            _pop_task("first_passing", parallel=False),
            tmp_path,
            "plan",
            None,
            "",
            "",
            None,
            None,
            None,
        )
        assert result.status == "failed"
        # The first candidate in the (sequential) order is returned.
        assert result.message == "haiku"

    def test_majority_without_quorum_returns_first_result(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Fewer than half succeed -> majority returns the first result."""

        def fake_execute(plan: PlanSpec, task: TaskSpec, run_path: Path, **kw: object) -> TaskResult:
            # Only one of two passes -> not > half.
            status = "success" if task.model == "sonnet" else "failed"
            return TaskResult(task_id=task.id, status=status, message=task.model)

        monkeypatch.setattr(runners, "execute_task", fake_execute)
        result = _run_population_search(
            _make_plan(),
            _pop_task("majority", parallel=False),
            tmp_path,
            "plan",
            None,
            "",
            "",
            None,
            None,
            None,
        )
        # First sequential candidate ('haiku', failed) is returned as fallback.
        assert result.message == "haiku"


# ---------------------------------------------------------------------------
# _build_layered_context — sanity: fitting loop with multiple upstreams
# ---------------------------------------------------------------------------


class TestBuildLayeredContextFitting:
    def test_small_budget_truncates_and_returns(self) -> None:
        """A small budget forces the fitting loop and still returns content."""
        out = _build_layered_context(
            {
                "u1": "first upstream content line that is reasonably long here",
                "u2": "second upstream content line that is also long enough now",
            },
            budget_tokens=20,
        )
        # Non-crashing string result within the (loosely enforced) budget.
        assert isinstance(out, str)
