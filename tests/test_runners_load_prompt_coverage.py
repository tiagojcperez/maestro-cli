from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path

import pytest

from maestro_cli.errors import TaskExecutionError
from maestro_cli.models import PlanDefaults, PlanSpec, TaskResult, TaskSpec
from maestro_cli.runners import _HONEYPOT_DECOYS, _load_prompt


def _make_plan(**kwargs: object) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="cov-plan",
        defaults=PlanDefaults(),
        tasks=[],
        **kwargs,  # type: ignore[arg-type]
    )


def _make_result(
    task_id: str,
    *,
    tainted: bool = False,
    stdout_tail: str = "upstream output line\n",
) -> TaskResult:
    now = datetime.now(UTC)
    return TaskResult(
        task_id=task_id,
        status="success",
        exit_code=0,
        started_at=now,
        finished_at=now,
        duration_sec=1.0,
        command="echo hi",
        log_path=Path("/tmp") / f"{task_id}.log",
        result_path=Path("/tmp") / f"{task_id}.result.json",
        stdout_tail=stdout_tail,
        tainted=tainted,
    )


class TestPromptMdFileMissing:
    """Covers L2902: prompt_md_file resolves to a path that does not exist."""

    def test_missing_md_file_raises_e100(self, tmp_path: Path) -> None:
        # Point at a markdown file that does not exist; _resolve_prompt_path
        # returns a path (via source_dir fallback) but .exists() is False.
        plan = _make_plan(source_path=tmp_path / "plan.yaml")
        task = TaskSpec(
            id="md-task",
            engine="claude",
            prompt_md_file="does_not_exist.md",
            prompt_md_heading="Prompt",
        )
        with pytest.raises(TaskExecutionError) as excinfo:
            _load_prompt(plan, task)
        assert "prompt_md_file not found" in str(excinfo.value)
        assert excinfo.value.code == "E100"

    def test_missing_md_file_with_no_source_dir(self) -> None:
        # No source_dir / workspace_root => _resolve_prompt_path may return
        # None for a relative path; still hits the not-found branch (L2902).
        plan = _make_plan()
        task = TaskSpec(
            id="md-task2",
            engine="claude",
            prompt_md_file="also_missing.md",
            prompt_md_heading="Prompt",
        )
        with pytest.raises(TaskExecutionError) as excinfo:
            _load_prompt(plan, task)
        assert "prompt_md_file not found" in str(excinfo.value)


class TestPromptMdExtractionFailure:
    """Covers L2909-2915: extract_prompt_from_markdown raises ValueError."""

    def test_heading_not_found_raises_e101(self, tmp_path: Path) -> None:
        md = tmp_path / "prompts.md"
        # File exists but does NOT contain the requested heading, so
        # extract_prompt_from_markdown raises ValueError -> E101.
        md.write_text("## Other Heading\n\n```text\nbody\n```\n", encoding="utf-8")
        plan = _make_plan(source_path=tmp_path / "plan.yaml")
        task = TaskSpec(
            id="md-bad-heading",
            engine="claude",
            prompt_md_file="prompts.md",
            prompt_md_heading="Nonexistent Heading",
        )
        with pytest.raises(TaskExecutionError) as excinfo:
            _load_prompt(plan, task)
        assert "markdown prompt extraction failed" in str(excinfo.value)
        assert excinfo.value.code == "E101"

    def test_valid_heading_extracts_successfully(self, tmp_path: Path) -> None:
        # Sanity path: a matching heading extracts and renders without error.
        md = tmp_path / "ok.md"
        md.write_text("## Prompt\n\n```text\nDo the thing\n```\n", encoding="utf-8")
        plan = _make_plan(source_path=tmp_path / "plan.yaml")
        task = TaskSpec(
            id="md-ok",
            engine="claude",
            prompt_md_file="ok.md",
            prompt_md_heading="Prompt",
        )
        result = _load_prompt(plan, task)
        assert "Do the thing" in result


class TestUpstreamResultNone:
    """Covers L2943: a context_from id maps to None in upstream_results."""

    def test_none_upstream_result_is_skipped(self) -> None:
        plan = _make_plan()
        task = TaskSpec(
            id="consumer",
            engine="claude",
            depends_on=["maybe"],
            context_from=["maybe"],
            prompt="Status: {{ maybe.status }}",
        )
        # The dict contains the key but its value is None -> `continue` (L2943).
        upstream: dict[str, TaskResult | None] = {"maybe": None}
        result = _load_prompt(plan, task, upstream)  # type: ignore[arg-type]
        # Because the result was skipped, no variable was set and the
        # placeholder remains unrendered.
        assert "{{ maybe.status }}" in result

    def test_mixed_none_and_present_upstream(self) -> None:
        plan = _make_plan()
        task = TaskSpec(
            id="consumer2",
            engine="claude",
            depends_on=["gone", "present"],
            context_from=["gone", "present"],
            prompt="A={{ gone.status }} B={{ present.status }}",
        )
        upstream: dict[str, TaskResult | None] = {
            "gone": None,
            "present": _make_result("present"),
        }
        result = _load_prompt(plan, task, upstream)  # type: ignore[arg-type]
        # `gone` skipped (L2943), `present` rendered.
        assert "{{ gone.status }}" in result
        assert "B=success" in result


class TestKnowledgeInjection:
    """Covers L3029: cross-run knowledge prepended when task_knowledge set."""

    def test_knowledge_prepended_for_engine_task(self) -> None:
        plan = _make_plan()
        task = TaskSpec(
            id="k-task",
            engine="claude",
            prompt="Implement the feature.",
        )
        result = _load_prompt(
            plan,
            task,
            None,
            extra_template_vars={"task_knowledge": "- watch out for timeouts (80%)"},
        )
        assert "## Previous Run Insights" in result
        assert "watch out for timeouts" in result
        assert "Implement the feature." in result
        # Knowledge section comes before the body.
        assert result.index("## Previous Run Insights") < result.index(
            "Implement the feature."
        )

    def test_no_knowledge_section_without_task_knowledge(self) -> None:
        plan = _make_plan()
        task = TaskSpec(id="k-task2", engine="claude", prompt="Body only.")
        result = _load_prompt(plan, task, None)
        assert "## Previous Run Insights" not in result

    def test_knowledge_skipped_when_no_engine(self) -> None:
        # task.engine is None -> the injection guard at L3028 is False even
        # though task_knowledge is present.
        plan = _make_plan()
        task = TaskSpec(id="k-cmd", command="echo hi", prompt="Body.")
        result = _load_prompt(
            plan,
            task,
            None,
            extra_template_vars={"task_knowledge": "- some lesson"},
        )
        assert "## Previous Run Insights" not in result


class TestHoneypotInjection:
    """Covers L3040-3044 and L3046: honeypot decoy injection."""

    def test_explicit_honeypot_flag_injects_decoys(self) -> None:
        # task.honeypot True -> _inject_honeypot True directly -> L3046.
        plan = _make_plan()
        task = TaskSpec(
            id="hp-explicit",
            engine="claude",
            prompt="Do work.",
            honeypot=True,
        )
        result = _load_prompt(plan, task, None)
        for decoy_name in _HONEYPOT_DECOYS:
            assert decoy_name in result

    def test_untrusted_tainted_upstream_triggers_honeypot(self) -> None:
        # honeypot flag is False, but context_trust=untrusted AND a consumed
        # upstream result is tainted -> loop L3040-3044 sets _inject_honeypot
        # and breaks, then L3046 injects decoys.
        plan = _make_plan()
        task = TaskSpec(
            id="hp-untrusted",
            engine="claude",
            depends_on=["src"],
            context_from=["src"],
            context_trust="untrusted",
            prompt="Summarize: {{ src.stdout_tail }}",
        )
        upstream = {"src": _make_result("src", tainted=True)}
        result = _load_prompt(plan, task, upstream)
        for decoy_name in _HONEYPOT_DECOYS:
            assert decoy_name in result

    def test_untrusted_but_not_tainted_no_honeypot(self) -> None:
        # context_trust=untrusted but the upstream is NOT tainted -> the loop
        # iterates without setting the flag (exercises the False branch of
        # L3042) and no decoys are injected.
        plan = _make_plan()
        task = TaskSpec(
            id="hp-clean",
            engine="claude",
            depends_on=["src"],
            context_from=["src"],
            context_trust="untrusted",
            prompt="Summarize: {{ src.stdout_tail }}",
        )
        upstream = {"src": _make_result("src", tainted=False)}
        result = _load_prompt(plan, task, upstream)
        for decoy_name in _HONEYPOT_DECOYS:
            assert decoy_name not in result

    def test_untrusted_none_upstream_no_honeypot(self) -> None:
        # context_trust=untrusted, context id maps to None -> the `_ur and ...`
        # short-circuit keeps honeypot off (L3042 falsy branch).
        plan = _make_plan()
        task = TaskSpec(
            id="hp-none",
            engine="claude",
            depends_on=["src"],
            context_from=["src"],
            context_trust="untrusted",
            prompt="Body {{ src.stdout_tail }}",
        )
        upstream: dict[str, TaskResult | None] = {"src": None}
        result = _load_prompt(plan, task, upstream)  # type: ignore[arg-type]
        for decoy_name in _HONEYPOT_DECOYS:
            assert decoy_name not in result
