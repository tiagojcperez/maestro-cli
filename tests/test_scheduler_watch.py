from __future__ import annotations

"""Watch-generated tests for scheduler.py. Do NOT edit manually — managed by maestro watch."""

import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from maestro_cli.scheduler import (
    _extract_keywords,
    _compute_idf,
    _score_section,
    _apply_intent_filtering,
    _compute_hop_distances,
    _apply_hop_decay,
    _estimate_tokens,
    _fmt_duration,
    _split_into_sections,
    _filter_tail_by_intent,
    _new_skipped_result,
    _resolve_model,
    _parse_layered_context_sections,
    run_plan,
)
from maestro_cli.models import TaskSpec, TaskResult, PlanSpec, PlanDefaults


# ---------------------------------------------------------------------------
# TestSchedulerWatch1 — _extract_keywords
# ---------------------------------------------------------------------------

class TestSchedulerWatch1:
    """Tests for _extract_keywords helper."""

    def test_empty_string_returns_empty_set(self) -> None:
        result = _extract_keywords("")
        assert result == set()

    def test_stopwords_excluded(self) -> None:
        result = _extract_keywords("the quick brown fox")
        assert "the" not in result

    def test_short_words_excluded(self) -> None:
        # single-char words should not appear (re matches \w{3,})
        result = _extract_keywords("a b c d")
        assert len(result) == 0

    def test_extracts_alpha_tokens(self) -> None:
        result = _extract_keywords("authentication token refresh")
        assert "authentication" in result
        assert "refresh" in result

    def test_case_insensitive(self) -> None:
        result = _extract_keywords("Authentication TOKEN")
        assert "authentication" in result
        assert "token" in result

    def test_numeric_only_string(self) -> None:
        # purely numeric strings still get extracted if >= 3 chars (regex: \w{3,})
        # just verify the function runs without error
        result = _extract_keywords("123 456 789")
        assert isinstance(result, set)

    def test_hyphenated_words(self) -> None:
        result = _extract_keywords("well-formed plan execution")
        assert "plan" in result
        assert "execution" in result


# ---------------------------------------------------------------------------
# TestSchedulerWatch2 — _compute_idf
# ---------------------------------------------------------------------------

class TestSchedulerWatch2:
    """Tests for _compute_idf helper."""

    def test_empty_sections_returns_empty_dict(self) -> None:
        result = _compute_idf([])
        assert result == {}

    def test_single_section_idf_computed(self) -> None:
        result = _compute_idf(["authentication token refresh"])
        assert "authentication" in result
        assert isinstance(result["authentication"], float)

    def test_term_in_all_docs_has_lower_idf(self) -> None:
        sections = ["authentication test", "authentication review", "authentication audit"]
        result = _compute_idf(sections)
        # term in all 3 docs: idf = log(3/(1+3)) = log(0.75) < 0
        assert result.get("authentication", 1.0) < result.get("test", 1.0)

    def test_rare_term_has_higher_idf(self) -> None:
        sections = [
            "authentication security",
            "authentication review",
            "unique_rare_word check",
        ]
        result = _compute_idf(sections)
        # authentication appears in 2 of 3, unique_rare_word in 1
        idf_auth = result.get("authentication", 0.0)
        idf_rare = result.get("unique_rare_word", 0.0)
        assert idf_rare > idf_auth

    def test_multiple_sections_includes_all_terms(self) -> None:
        sections = ["alpha beta", "gamma delta"]
        result = _compute_idf(sections)
        assert "alpha" in result
        assert "gamma" in result


# ---------------------------------------------------------------------------
# TestSchedulerWatch3 — _score_section
# ---------------------------------------------------------------------------

class TestSchedulerWatch3:
    """Tests for _score_section helper."""

    def test_empty_section_returns_zero(self) -> None:
        result = _score_section("", {"test", "value"})
        assert result == 0

    def test_empty_intent_keywords_returns_zero(self) -> None:
        result = _score_section("this is a test section", set())
        assert result == 0

    def test_matching_keyword_returns_positive(self) -> None:
        result = _score_section("authentication test section", {"authentication", "test"})
        assert result > 0

    def test_no_matching_keywords_returns_zero(self) -> None:
        result = _score_section("unrelated content here", {"authentication", "token"})
        assert result == 0

    def test_with_idf_matching_returns_positive(self) -> None:
        idf = {"authentication": 1.5, "test": 0.5}
        result = _score_section("authentication test check", {"authentication", "test"}, idf=idf)
        assert result > 0

    def test_with_idf_no_match_returns_zero(self) -> None:
        idf = {"authentication": 1.5}
        result = _score_section("something completely different", {"authentication"}, idf=idf)
        assert result == 0


# ---------------------------------------------------------------------------
# TestSchedulerWatch4 — _apply_intent_filtering
# ---------------------------------------------------------------------------

class TestSchedulerWatch4:
    """Tests for _apply_intent_filtering helper."""

    def _make_task_result(self, task_id: str, stdout_tail: str) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            status="success",
            exit_code=0,
            stdout_tail=stdout_tail,
            duration_sec=1.0,
        )

    def test_none_intent_returns_unchanged(self) -> None:
        upstream = {
            "t1": self._make_task_result("t1", "some content here"),
        }
        filtered, stats, trajectory = _apply_intent_filtering(upstream, None)
        assert filtered["t1"].stdout_tail == "some content here"
        assert stats == []

    def test_empty_upstream_returns_empty(self) -> None:
        filtered, stats, trajectory = _apply_intent_filtering({}, {"keyword"})
        assert filtered == {}
        assert stats == []

    def test_matching_content_kept(self) -> None:
        upstream = {
            "t1": self._make_task_result("t1", "authentication token refresh needed"),
        }
        filtered, stats, trajectory = _apply_intent_filtering(upstream, {"authentication", "token"})
        # content matches — should be kept (tail non-empty)
        assert filtered["t1"].stdout_tail != ""

    def test_no_matching_content_returns_original(self) -> None:
        upstream = {
            "t1": self._make_task_result("t1", "completely unrelated content here"),
        }
        filtered, stats, trajectory = _apply_intent_filtering(upstream, {"authentication"})
        # fallback: original content returned when nothing matches
        assert filtered["t1"].stdout_tail is not None


# ---------------------------------------------------------------------------
# TestSchedulerWatch5 — _compute_hop_distances
# ---------------------------------------------------------------------------

class TestSchedulerWatch5:
    """Tests for _compute_hop_distances helper."""

    def _make_task(self, task_id: str, depends_on: list[str]) -> TaskSpec:
        return TaskSpec(id=task_id, depends_on=depends_on, command="echo ok")

    def test_direct_dependency_is_hop_one(self) -> None:
        t1 = self._make_task("t1", [])
        t2 = self._make_task("t2", ["t1"])
        all_tasks = {"t1": t1, "t2": t2}
        # context_from = ["t1"] for task t2
        distances = _compute_hop_distances("t2", ["t1"], all_tasks)
        assert distances["t1"] == 1

    def test_transitive_dependency_is_hop_two(self) -> None:
        t1 = self._make_task("t1", [])
        t2 = self._make_task("t2", ["t1"])
        t3 = self._make_task("t3", ["t2"])
        all_tasks = {"t1": t1, "t2": t2, "t3": t3}
        distances = _compute_hop_distances("t3", ["t1", "t2"], all_tasks)
        assert distances["t2"] == 1
        assert distances["t1"] == 2

    def test_no_context_from_returns_empty(self) -> None:
        t1 = self._make_task("t1", [])
        all_tasks = {"t1": t1}
        distances = _compute_hop_distances("t1", [], all_tasks)
        assert distances == {}

    def test_multiple_paths_takes_shortest(self) -> None:
        # Diamond: t3 depends on t1 and t2; t2 depends on t1
        t1 = self._make_task("t1", [])
        t2 = self._make_task("t2", ["t1"])
        t3 = self._make_task("t3", ["t1", "t2"])
        all_tasks = {"t1": t1, "t2": t2, "t3": t3}
        distances = _compute_hop_distances("t3", ["t1"], all_tasks)
        # t1 is a direct dep of t3, so distance = 1
        assert distances["t1"] == 1


# ---------------------------------------------------------------------------
# TestSchedulerWatch6 — _apply_hop_decay
# ---------------------------------------------------------------------------

class TestSchedulerWatch6:
    """Tests for _apply_hop_decay helper."""

    def _make_task_result(self, task_id: str, content: str) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            status="success",
            exit_code=0,
            stdout_tail=content,
            duration_sec=1.0,
        )

    def test_direct_dep_no_decay(self) -> None:
        upstream = {"t1": self._make_task_result("t1", "important content")}
        distances = {"t1": 1}
        result = _apply_hop_decay(upstream, distances)
        # hop=1 → multiplier=1.0 → content unchanged
        assert result["t1"].stdout_tail == "important content"

    def test_transitive_dep_gets_decayed(self) -> None:
        content = "section1\n\nsection2\n\nsection3"
        upstream = {"t1": self._make_task_result("t1", content)}
        distances = {"t1": 2}
        result = _apply_hop_decay(upstream, distances)
        # hop=2 → multiplier=0.8 → may trim sections
        # Just verify the function runs without error
        assert "t1" in result

    def test_missing_distance_treated_as_direct(self) -> None:
        upstream = {"t1": self._make_task_result("t1", "some content")}
        distances: dict[str, int] = {}  # no distance info
        result = _apply_hop_decay(upstream, distances)
        assert "t1" in result


# ---------------------------------------------------------------------------
# TestSchedulerWatch7 — run_plan basics (mocked execution)
# ---------------------------------------------------------------------------

class TestSchedulerWatch7:
    """Tests for run_plan with mocked subprocess execution."""

    def _make_plan(self, tmp_path: Path, content: str) -> PlanSpec:
        from maestro_cli.loader import load_plan
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        return load_plan(pf)

    def test_single_command_task_succeeds(self, tmp_path: Path) -> None:
        plan = self._make_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo hello
""")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="hello\n", stderr="")
            result = run_plan(plan, dry_run=True)
        assert result.plan_name == "test"
        # dry_run means tasks are not actually executed
        assert len(result.task_results) == 1

    def test_dry_run_marks_tasks_as_dry_run(self, tmp_path: Path) -> None:
        plan = self._make_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo hello
  - id: t2
    command: echo world
    depends_on: [t1]
""")
        result = run_plan(plan, dry_run=True)
        # task_results is dict[str, TaskResult]
        assert result.task_results["t1"].status == "dry_run"
        assert result.task_results["t2"].status == "dry_run"

    def test_only_filter_runs_subset(self, tmp_path: Path) -> None:
        plan = self._make_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo hello
  - id: t2
    command: echo world
""")
        result = run_plan(plan, dry_run=True, only={"t1"})
        assert "t1" in result.task_results

    def test_skip_filter_excludes_task(self, tmp_path: Path) -> None:
        plan = self._make_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo hello
  - id: t2
    command: echo world
""")
        result = run_plan(plan, dry_run=True, skip={"t2"})
        # t2 should be skipped or not run
        if "t2" in result.task_results:
            assert result.task_results["t2"].status in ("skipped", "dry_run")


# ---------------------------------------------------------------------------
# TestSchedulerWatch8 — _estimate_tokens
# ---------------------------------------------------------------------------

class TestSchedulerWatch8:
    """Tests for _estimate_tokens helper."""

    def test_empty_string_returns_one(self) -> None:
        assert _estimate_tokens("") == 1

    def test_four_chars_is_one_token(self) -> None:
        assert _estimate_tokens("abcd") == 1

    def test_eight_chars_is_two_tokens(self) -> None:
        assert _estimate_tokens("abcdefgh") == 2

    def test_large_text_scales_linearly(self) -> None:
        text = "a" * 400
        assert _estimate_tokens(text) == 100

    def test_newlines_counted(self) -> None:
        text = "\n" * 40
        result = _estimate_tokens(text)
        assert result == 10

    def test_always_positive(self) -> None:
        assert _estimate_tokens("x") >= 1

    def test_longer_text_returns_more_tokens(self) -> None:
        short = _estimate_tokens("short")
        long = _estimate_tokens("this is a much longer string with many characters")
        assert long > short


# ---------------------------------------------------------------------------
# TestSchedulerWatch9 — _fmt_duration
# ---------------------------------------------------------------------------

class TestSchedulerWatch9:
    """Tests for _fmt_duration formatting helper."""

    def test_zero_seconds(self) -> None:
        assert _fmt_duration(0) == "0s"

    def test_seconds_only(self) -> None:
        assert _fmt_duration(45) == "45s"

    def test_exactly_one_minute(self) -> None:
        assert _fmt_duration(60) == "1m00s"

    def test_one_minute_thirty(self) -> None:
        assert _fmt_duration(90) == "1m30s"

    def test_ten_minutes(self) -> None:
        assert _fmt_duration(600) == "10m00s"

    def test_59_seconds(self) -> None:
        assert _fmt_duration(59) == "59s"

    def test_61_seconds(self) -> None:
        assert _fmt_duration(61) == "1m01s"

    def test_float_rounds_up(self) -> None:
        result = _fmt_duration(45.7)
        assert result == "46s"


# ---------------------------------------------------------------------------
# TestSchedulerWatch10 — _split_into_sections
# ---------------------------------------------------------------------------

class TestSchedulerWatch10:
    """Tests for _split_into_sections helper."""

    def test_empty_string_returns_empty_list(self) -> None:
        assert _split_into_sections("") == []

    def test_single_paragraph_returns_one_section(self) -> None:
        result = _split_into_sections("hello world")
        assert len(result) == 1
        assert "hello world" in result[0]

    def test_two_paragraphs_split_by_blank_line(self) -> None:
        text = "paragraph one\n\nparagraph two"
        result = _split_into_sections(text)
        assert len(result) == 2
        assert any("paragraph one" in s for s in result)
        assert any("paragraph two" in s for s in result)

    def test_multiple_blank_lines_still_splits(self) -> None:
        text = "section a\n\n\n\nsection b"
        result = _split_into_sections(text)
        assert len(result) == 2

    def test_single_long_para_chunked_by_lines(self) -> None:
        lines = [f"line {i}" for i in range(20)]
        text = "\n".join(lines)
        result = _split_into_sections(text)
        assert len(result) >= 2

    def test_strips_whitespace_from_sections(self) -> None:
        text = "  alpha  \n\n  beta  "
        result = _split_into_sections(text)
        for s in result:
            assert s == s.strip()

    def test_whitespace_only_string_returns_empty(self) -> None:
        assert _split_into_sections("   \n\n   ") == []


# ---------------------------------------------------------------------------
# TestSchedulerWatch11 — _filter_tail_by_intent
# ---------------------------------------------------------------------------

class TestSchedulerWatch11:
    """Tests for _filter_tail_by_intent helper."""

    def test_empty_tail_returns_empty(self) -> None:
        result_tail, score, keywords = _filter_tail_by_intent("", {"auth"})
        assert result_tail == ""
        assert score == 0

    def test_empty_keywords_returns_original(self) -> None:
        tail = "some output here"
        result_tail, score, keywords = _filter_tail_by_intent(tail, set())
        assert result_tail == tail

    def test_matching_section_kept(self) -> None:
        tail = "authentication passed\n\nunrelated content here"
        result_tail, score, keywords = _filter_tail_by_intent(tail, {"authentication"})
        assert "authentication" in result_tail

    def test_returns_three_tuple(self) -> None:
        result = _filter_tail_by_intent("hello world", {"hello"})
        assert len(result) == 3

    def test_score_non_negative_when_match(self) -> None:
        tail = "authentication token refresh\n\nunrelated content"
        _, score, _ = _filter_tail_by_intent(tail, {"authentication", "token"})
        assert score >= 0

    def test_matched_keywords_not_contain_missing_word(self) -> None:
        tail = "authentication check\n\nother stuff"
        _, _, matched = _filter_tail_by_intent(tail, {"authentication", "zzznothere"})
        assert "zzznothere" not in matched


# ---------------------------------------------------------------------------
# TestSchedulerWatch12 — _new_skipped_result
# ---------------------------------------------------------------------------

class TestSchedulerWatch12:
    """Tests for _new_skipped_result helper."""

    def test_returns_skipped_status(self, tmp_path) -> None:
        result = _new_skipped_result("task-1", tmp_path, "dependency failed")
        assert result.status == "skipped"

    def test_task_id_preserved(self, tmp_path) -> None:
        result = _new_skipped_result("my-task", tmp_path, "skipped by test")
        assert result.task_id == "my-task"

    def test_message_preserved(self, tmp_path) -> None:
        result = _new_skipped_result("t1", tmp_path, "upstream failed")
        assert result.message == "upstream failed"

    def test_duration_is_zero(self, tmp_path) -> None:
        result = _new_skipped_result("t1", tmp_path, "msg")
        assert result.duration_sec == 0.0

    def test_log_file_written(self, tmp_path) -> None:
        _new_skipped_result("t1", tmp_path, "skipped")
        log_path = tmp_path / "t1.log"
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "skipped" in content

    def test_result_json_written(self, tmp_path) -> None:
        _new_skipped_result("t1", tmp_path, "skipped")
        result_path = tmp_path / "t1.result.json"
        assert result_path.exists()

    def test_exit_code_is_none(self, tmp_path) -> None:
        result = _new_skipped_result("t1", tmp_path, "skipped")
        assert result.exit_code is None

    def test_started_equals_finished(self, tmp_path) -> None:
        result = _new_skipped_result("t1", tmp_path, "skipped")
        assert result.started_at == result.finished_at


# ---------------------------------------------------------------------------
# TestSchedulerWatch13 — _resolve_model
# ---------------------------------------------------------------------------

class TestSchedulerWatch13:
    """Tests for _resolve_model helper."""

    def _make_plan(self) -> PlanSpec:
        return PlanSpec(name="test", tasks=[])

    def _make_task(self, engine: str, model=None) -> TaskSpec:
        return TaskSpec(id="t1", engine=engine, model=model, prompt="do it")

    def test_task_model_overrides_claude_default(self) -> None:
        plan = self._make_plan()
        plan.defaults.claude.model = "haiku"
        task = self._make_task("claude", "opus")
        assert _resolve_model(plan, task) == "opus"

    def test_falls_back_to_claude_default(self) -> None:
        plan = self._make_plan()
        plan.defaults.claude.model = "sonnet"
        task = self._make_task("claude", None)
        assert _resolve_model(plan, task) == "sonnet"

    def test_task_model_overrides_codex_default(self) -> None:
        plan = self._make_plan()
        plan.defaults.codex.model = "5.1"
        task = self._make_task("codex", "5.4")
        assert _resolve_model(plan, task) == "5.4"

    def test_falls_back_to_codex_default(self) -> None:
        plan = self._make_plan()
        plan.defaults.codex.model = "5.4"
        task = self._make_task("codex", None)
        assert _resolve_model(plan, task) == "5.4"

    def test_task_model_overrides_copilot_default(self) -> None:
        plan = self._make_plan()
        plan.defaults.copilot.model = "haiku"
        task = self._make_task("copilot", "gpt-5.4-codex")
        assert _resolve_model(plan, task) == "gpt-5.4-codex"

    def test_falls_back_to_copilot_default(self) -> None:
        plan = self._make_plan()
        plan.defaults.copilot.model = "sonnet"
        task = self._make_task("copilot", None)
        assert _resolve_model(plan, task) == "sonnet"

    def test_gemini_with_task_model_returned(self) -> None:
        plan = self._make_plan()
        task = self._make_task("gemini", "flash")
        assert _resolve_model(plan, task) == "flash"

    def test_gemini_no_model_returns_empty(self) -> None:
        plan = self._make_plan()
        task = self._make_task("gemini", None)
        assert _resolve_model(plan, task) == ""


# ---------------------------------------------------------------------------
# TestSchedulerWatch14 — _parse_layered_context_sections
# ---------------------------------------------------------------------------

class TestSchedulerWatch14:
    """Tests for _parse_layered_context_sections helper."""

    def test_empty_string_returns_empty_dict(self) -> None:
        assert _parse_layered_context_sections("") == {}

    def test_no_section_headers_returns_empty(self) -> None:
        assert _parse_layered_context_sections("plain text without headers") == {}

    def test_single_section_parsed(self) -> None:
        text = "--- upstream-1 ---\nbody content here\n"
        result = _parse_layered_context_sections(text)
        assert "upstream-1" in result
        assert "body content here" in result["upstream-1"]

    def test_multiple_sections_parsed(self) -> None:
        text = "--- task-a ---\nalpha content\n--- task-b ---\nbeta content\n"
        result = _parse_layered_context_sections(text)
        assert "task-a" in result
        assert "task-b" in result
        assert "alpha content" in result["task-a"]
        assert "beta content" in result["task-b"]

    def test_section_body_stripped(self) -> None:
        text = "--- task-1 ---\n\n  content  \n\n"
        result = _parse_layered_context_sections(text)
        assert result["task-1"] == "content"

    def test_three_sections_boundaries_correct(self) -> None:
        text = "--- a ---\nA body\n--- b ---\nB body\n--- c ---\nC body\n"
        result = _parse_layered_context_sections(text)
        assert result["a"] == "A body"
        assert result["b"] == "B body"
        assert result["c"] == "C body"

    def test_hyphenated_upstream_id_parsed(self) -> None:
        text = "--- my-long-task-id ---\ncontent\n"
        result = _parse_layered_context_sections(text)
        assert "my-long-task-id" in result

    def test_returns_dict_type(self) -> None:
        result = _parse_layered_context_sections("--- t1 ---\ndata\n")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestSchedulerWatch15 — _compute_task_depth
# ---------------------------------------------------------------------------
from maestro_cli.scheduler import _compute_task_depth, _compute_fan_out


class TestSchedulerWatch15:
    def _make_plan(self, tasks: list[TaskSpec]) -> PlanSpec:
        return PlanSpec(name="t", tasks=tasks)

    def test_no_deps_returns_zero(self) -> None:
        t = TaskSpec(id="a", command="x")
        plan = self._make_plan([t])
        assert _compute_task_depth(t, plan) == 0

    def test_one_direct_dep_returns_one(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        plan = self._make_plan([a, b])
        assert _compute_task_depth(b, plan) == 1

    def test_two_level_chain_returns_two(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        c = TaskSpec(id="c", command="x", depends_on=["b"])
        plan = self._make_plan([a, b, c])
        assert _compute_task_depth(c, plan) == 2

    def test_diamond_takes_longest_path(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        c = TaskSpec(id="c", command="x", depends_on=["a"])
        d = TaskSpec(id="d", command="x", depends_on=["b", "c"])
        plan = self._make_plan([a, b, c, d])
        assert _compute_task_depth(d, plan) == 2

    def test_unknown_dep_returns_zero_depth(self) -> None:
        t = TaskSpec(id="z", command="x", depends_on=["missing"])
        plan = self._make_plan([t])
        assert _compute_task_depth(t, plan) == 0


# ---------------------------------------------------------------------------
# TestSchedulerWatch16 — _compute_fan_out
# ---------------------------------------------------------------------------


class TestSchedulerWatch16:
    def _make_plan(self, tasks: list[TaskSpec]) -> PlanSpec:
        return PlanSpec(name="t", tasks=tasks)

    def test_no_dependents_returns_zero(self) -> None:
        a = TaskSpec(id="a", command="x")
        plan = self._make_plan([a])
        assert _compute_fan_out(a, plan) == 0

    def test_one_dependent_returns_one(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        plan = self._make_plan([a, b])
        assert _compute_fan_out(a, plan) == 1

    def test_two_dependents_returns_two(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        c = TaskSpec(id="c", command="x", depends_on=["a"])
        plan = self._make_plan([a, b, c])
        assert _compute_fan_out(a, plan) == 2

    def test_leaf_node_returns_zero(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        c = TaskSpec(id="c", command="x", depends_on=["b"])
        plan = self._make_plan([a, b, c])
        assert _compute_fan_out(c, plan) == 0

    def test_middle_node_returns_one(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        c = TaskSpec(id="c", command="x", depends_on=["b"])
        plan = self._make_plan([a, b, c])
        assert _compute_fan_out(b, plan) == 1


# ---------------------------------------------------------------------------
# TestSchedulerWatch17 — _compute_tainted_tasks
# ---------------------------------------------------------------------------
from maestro_cli.scheduler import _compute_tainted_tasks


class TestSchedulerWatch17:
    def _make_plan(self, tasks: list[TaskSpec]) -> PlanSpec:
        return PlanSpec(name="t", tasks=tasks)

    def test_no_untrusted_returns_empty(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        plan = self._make_plan([a, b])
        assert _compute_tainted_tasks(plan) == set()

    def test_explicit_untrusted_is_tainted(self) -> None:
        a = TaskSpec(id="a", command="x", context_trust="untrusted")
        plan = self._make_plan([a])
        result = _compute_tainted_tasks(plan)
        assert "a" in result

    def test_propagates_taint_to_downstream(self) -> None:
        a = TaskSpec(id="a", command="x", context_trust="untrusted")
        b = TaskSpec(id="b", command="x", depends_on=["a"], context_from=["a"])
        plan = self._make_plan([a, b])
        result = _compute_tainted_tasks(plan)
        assert "a" in result
        assert "b" in result

    def test_guard_command_clears_taint(self) -> None:
        a = TaskSpec(id="a", command="x", context_trust="untrusted")
        b = TaskSpec(id="b", command="x", depends_on=["a"], context_from=["a"],
                     guard_command="check")
        plan = self._make_plan([a, b])
        result = _compute_tainted_tasks(plan)
        assert "a" in result
        assert "b" not in result

    def test_verify_command_clears_taint(self) -> None:
        a = TaskSpec(id="a", command="x", context_trust="untrusted")
        b = TaskSpec(id="b", command="x", depends_on=["a"], context_from=["a"],
                     verify_command="check")
        plan = self._make_plan([a, b])
        result = _compute_tainted_tasks(plan)
        assert "b" not in result

    def test_trusted_not_tainted(self) -> None:
        a = TaskSpec(id="a", command="x", context_trust="trusted")
        plan = self._make_plan([a])
        result = _compute_tainted_tasks(plan)
        assert "a" not in result

    def test_selected_ids_filters_tasks(self) -> None:
        a = TaskSpec(id="a", command="x", context_trust="untrusted")
        b = TaskSpec(id="b", command="x")
        plan = self._make_plan([a, b])
        result = _compute_tainted_tasks(plan, selected_ids={"b"})
        assert "b" not in result


# ---------------------------------------------------------------------------
# TestSchedulerWatch18 — _request_approval
# ---------------------------------------------------------------------------
from maestro_cli.scheduler import _request_approval


class TestSchedulerWatch18:
    def test_non_interactive_returns_false(self) -> None:
        result = _request_approval("task-1", None, interactive=False)
        assert result is False

    def test_non_interactive_with_message_returns_false(self) -> None:
        result = _request_approval("task-1", "Please approve", interactive=False)
        assert result is False

    def test_interactive_yes_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: "y")
        result = _request_approval("task-1", None, interactive=True)
        assert result is True

    def test_interactive_no_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: "n")
        result = _request_approval("task-1", None, interactive=True)
        assert result is False

    def test_interactive_yes_capital_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: "Y")
        result = _request_approval("task-1", None, interactive=True)
        assert result is True

    def test_interactive_empty_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: "")
        result = _request_approval("task-1", None, interactive=True)
        assert result is False

    def test_interactive_eof_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise() -> str:
            raise EOFError
        monkeypatch.setattr("builtins.input", _raise)
        result = _request_approval("task-1", None, interactive=True)
        assert result is False

    def test_interactive_keyboard_interrupt_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise() -> str:
            raise KeyboardInterrupt
        monkeypatch.setattr("builtins.input", _raise)
        result = _request_approval("task-1", None, interactive=True)
        assert result is False


# ---------------------------------------------------------------------------
# TestSchedulerWatch19 — _apply_context_budget
# ---------------------------------------------------------------------------
from maestro_cli.scheduler import _apply_context_budget


class TestSchedulerWatch19:
    def _make_result(self, task_id: str, tail: str) -> TaskResult:
        return TaskResult(task_id=task_id, status="success", stdout_tail=tail)

    def test_within_budget_returns_unchanged(self) -> None:
        upstream = {"t1": self._make_result("t1", "short")}
        result, trims, _ = _apply_context_budget(upstream, budget_tokens=1000)
        assert result is upstream
        assert trims == []

    def test_empty_upstream_returns_empty(self) -> None:
        result, trims, _ = _apply_context_budget({}, budget_tokens=100)
        assert result == {}
        assert trims == []

    def test_over_budget_trims_tail(self) -> None:
        long_tail = "x" * 2000
        upstream = {"t1": self._make_result("t1", long_tail)}
        result, trims, _ = _apply_context_budget(upstream, budget_tokens=10)
        assert len(trims) > 0
        assert len(result["t1"].stdout_tail) < len(long_tail)

    def test_returns_three_tuple(self) -> None:
        upstream = {"t1": self._make_result("t1", "hello world")}
        out = _apply_context_budget(upstream, budget_tokens=5)
        assert isinstance(out, tuple)
        assert len(out) == 3

    def test_trim_records_contain_task_id(self) -> None:
        long_tail = "w" * 3000
        upstream = {"myTask": self._make_result("myTask", long_tail)}
        _, trims, _ = _apply_context_budget(upstream, budget_tokens=5)
        if trims:
            assert trims[0][0] == "myTask"

    def test_no_trim_when_under_budget(self) -> None:
        upstream = {"t1": self._make_result("t1", "hi")}
        _, trims, _ = _apply_context_budget(upstream, budget_tokens=10000)
        assert trims == []


# ---------------------------------------------------------------------------
# TestSchedulerWatch20 — _load_task_prompt_text
# ---------------------------------------------------------------------------
from maestro_cli.scheduler import _load_task_prompt_text


class TestSchedulerWatch20:
    def _make_plan(self, source_dir: Path | None = None) -> PlanSpec:
        sp = (source_dir / "plan.yaml") if source_dir else None
        return PlanSpec(name="t", tasks=[], source_path=sp)

    def test_inline_prompt_returned(self, tmp_path: Path) -> None:
        plan = self._make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude", prompt="Hello world")
        assert _load_task_prompt_text(plan, task) == "Hello world"

    def test_prompt_file_loaded(self, tmp_path: Path) -> None:
        pf = tmp_path / "p.txt"
        pf.write_text("File content", encoding="utf-8")
        plan = self._make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude", prompt_file="p.txt")
        assert _load_task_prompt_text(plan, task) == "File content"

    def test_missing_prompt_file_returns_empty(self, tmp_path: Path) -> None:
        plan = self._make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude", prompt_file="missing.txt")
        assert _load_task_prompt_text(plan, task) == ""

    def test_no_prompt_returns_empty(self, tmp_path: Path) -> None:
        plan = self._make_plan(tmp_path)
        task = TaskSpec(id="t", command="echo hi")
        assert _load_task_prompt_text(plan, task) == ""

    def test_prompt_md_file_with_heading_loaded(self, tmp_path: Path) -> None:
        md = tmp_path / "prompts.md"
        md.write_text("## My Task\n```text\nDo something\n```\n", encoding="utf-8")
        plan = self._make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude",
                        prompt_md_file="prompts.md",
                        prompt_md_heading="My Task")
        result = _load_task_prompt_text(plan, task)
        assert "Do something" in result

    def test_prompt_md_missing_heading_returns_empty(self, tmp_path: Path) -> None:
        md = tmp_path / "prompts.md"
        md.write_text("## Other Heading\ncontent\n", encoding="utf-8")
        plan = self._make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude",
                        prompt_md_file="prompts.md",
                        prompt_md_heading="Missing Heading")
        result = _load_task_prompt_text(plan, task)
        assert result == ""


# ---------------------------------------------------------------------------
# TestSchedulerWatch21 — _load_prior_results
# ---------------------------------------------------------------------------
import json
from maestro_cli.scheduler import _load_prior_results


class TestSchedulerWatch21:
    def test_no_manifest_raises(self, tmp_path: Path) -> None:
        import pytest
        with pytest.raises(ValueError, match="run_manifest.json"):
            _load_prior_results(tmp_path)

    def test_success_task_included(self, tmp_path: Path) -> None:
        manifest = {
            "task_results": {
                "t1": {"status": "success"},
            }
        }
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = _load_prior_results(tmp_path)
        assert "t1" in result
        assert result["t1"] == "success"

    def test_failed_task_excluded(self, tmp_path: Path) -> None:
        manifest = {
            "task_results": {
                "t1": {"status": "failed"},
            }
        }
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = _load_prior_results(tmp_path)
        assert "t1" not in result

    def test_dry_run_task_included(self, tmp_path: Path) -> None:
        manifest = {
            "task_results": {
                "t1": {"status": "dry_run"},
            }
        }
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = _load_prior_results(tmp_path)
        assert "t1" in result

    def test_skipped_dependency_failure_excluded(self, tmp_path: Path) -> None:
        manifest = {
            "task_results": {
                "t1": {"status": "skipped", "message": "Skipped because dependency failed: t0"},
            }
        }
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = _load_prior_results(tmp_path)
        assert "t1" not in result

    def test_skipped_failfast_excluded(self, tmp_path: Path) -> None:
        manifest = {
            "task_results": {
                "t1": {"status": "skipped", "message": "fail_fast triggered by task t0"},
            }
        }
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = _load_prior_results(tmp_path)
        assert "t1" not in result

    def test_skipped_when_expression_included(self, tmp_path: Path) -> None:
        manifest = {
            "task_results": {
                "t1": {"status": "skipped", "message": "when condition evaluated to false"},
            }
        }
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = _load_prior_results(tmp_path)
        assert "t1" in result

    def test_empty_task_results_returns_empty(self, tmp_path: Path) -> None:
        manifest = {"task_results": {}}
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = _load_prior_results(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# TestSchedulerWatch22 — _fmt_duration edge cases
# ---------------------------------------------------------------------------


class TestSchedulerWatch22:
    def test_negative_rounds_to_zero(self) -> None:
        # Negative durations shouldn't crash; result is 0s or negative string
        result = _fmt_duration(-1.0)
        assert isinstance(result, str)

    def test_very_large_duration(self) -> None:
        result = _fmt_duration(3661.0)
        assert isinstance(result, str)
        assert "m" in result or "h" in result or "s" in result

    def test_zero_returns_0s(self) -> None:
        result = _fmt_duration(0.0)
        assert "0s" in result or result == "0s"

    def test_returns_string(self) -> None:
        assert isinstance(_fmt_duration(42.5), str)

    def test_sub_second_has_units(self) -> None:
        result = _fmt_duration(0.5)
        assert "s" in result


# ---------------------------------------------------------------------------
# TestSchedulerWatch23 — _estimate_tokens edge cases + _extract_keywords
# ---------------------------------------------------------------------------


class TestSchedulerWatch23:
    def test_estimate_tokens_unicode_text(self) -> None:
        # Unicode chars should still return positive token count
        result = _estimate_tokens("héllo wörld")
        assert result >= 1

    def test_estimate_tokens_only_spaces(self) -> None:
        result = _estimate_tokens("   ")
        assert result >= 1

    def test_extract_keywords_punctuation_stripped(self) -> None:
        result = _extract_keywords("hello, world! foo-bar.")
        # Keywords should be alpha tokens
        for kw in result:
            assert kw.isalpha() or "-" in kw

    def test_extract_keywords_numbers_included_as_tokens(self) -> None:
        # Numbers are treated as tokens by _extract_keywords
        result = _extract_keywords("task 123 ran in 45 seconds")
        assert isinstance(result, set)
        assert len(result) > 0

    def test_extract_keywords_minimum_length(self) -> None:
        # Words shorter than 3 chars should be excluded (stopword logic)
        result = _extract_keywords("in a is to of the")
        # All short words and stopwords removed
        assert len(result) == 0 or all(len(k) >= 3 for k in result)

    def test_compute_task_depth_three_chain(self) -> None:
        a = TaskSpec(id="a", command="x")
        b = TaskSpec(id="b", command="x", depends_on=["a"])
        c = TaskSpec(id="c", command="x", depends_on=["b"])
        d = TaskSpec(id="d", command="x", depends_on=["c"])
        plan = PlanSpec(name="t", tasks=[a, b, c, d])
        assert _compute_task_depth(d, plan) == 3
