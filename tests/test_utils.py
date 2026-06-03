from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.utils import (
    command_to_string,
    extract_prompt_from_markdown,
    humanize_output_line,
    now_utc,
    render_template,
    resolve_path,
    sanitize_dirname,
)


class TestNowUtc:
    def test_returns_datetime_with_utc(self) -> None:
        from datetime import UTC

        result = now_utc()
        assert result.tzinfo is not None
        assert result.tzinfo == UTC


class TestResolvePath:
    def test_none_returns_none(self, tmp_path: Path) -> None:
        assert resolve_path(tmp_path, None) is None

    def test_empty_string_returns_none(self, tmp_path: Path) -> None:
        assert resolve_path(tmp_path, "") is None

    def test_absolute_path_returned_as_is(self, tmp_path: Path) -> None:
        abs_path = str(tmp_path / "some_file.txt")
        result = resolve_path(tmp_path, abs_path)
        assert result == Path(abs_path)

    def test_relative_path_joined_to_base(self, tmp_path: Path) -> None:
        result = resolve_path(tmp_path, "sub/file.txt")
        assert result is not None
        assert result.is_absolute()
        assert str(tmp_path) in str(result)
        assert result.name == "file.txt"


class TestCommandToString:
    def test_string_passed_through(self) -> None:
        assert command_to_string("echo hello") == "echo hello"

    def test_list_formatted(self) -> None:
        result = command_to_string(["echo", "hello", "world"])
        assert "echo" in result
        assert "hello" in result
        assert "world" in result

    def test_list_with_spaces_in_args(self) -> None:
        result = command_to_string(["echo", "hello world"])
        # The argument with a space should be quoted or escaped
        assert "hello world" in result or "hello\\ world" in result or '"hello world"' in result


class TestExtractPromptFromMarkdown:
    def test_valid_extraction(self) -> None:
        md = """\
## My Prompt

Some description text.

```text
Do this thing.
And also this.
```
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert result == "Do this thing.\nAnd also this.\n"

    def test_heading_not_found_raises(self) -> None:
        md = """\
## Other Heading

```text
Some content.
```
"""
        with pytest.raises(ValueError, match="Heading not found"):
            extract_prompt_from_markdown(md, "Missing Heading")

    def test_prose_fallback_when_no_fence(self) -> None:
        md = """\
## My Prompt

Just plain text, no code fence.
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert result == "Just plain text, no code fence.\n"

    def test_prose_fallback_multiline(self) -> None:
        md = """\
## integration

Review the codebase and ensure all modules integrate correctly.
Check for import errors and circular dependencies.
Verify that the test suite passes.

## next-section

Something else.
"""
        result = extract_prompt_from_markdown(md, "integration")
        assert "Review the codebase" in result
        assert "Verify that the test suite" in result
        assert "Something else" not in result

    def test_empty_prose_section_raises(self) -> None:
        md = """\
## My Prompt

## Next Section

Content here.
"""
        with pytest.raises(ValueError, match="Empty section"):
            extract_prompt_from_markdown(md, "My Prompt")

    def test_unclosed_fence_raises(self) -> None:
        md = """\
## My Prompt

```text
This fence is never closed.
"""
        with pytest.raises(ValueError, match="Unclosed code fence"):
            extract_prompt_from_markdown(md, "My Prompt")

    def test_empty_fence_raises(self) -> None:
        md = """\
## My Prompt

```text
```
"""
        with pytest.raises(ValueError, match="Empty prompt block"):
            extract_prompt_from_markdown(md, "My Prompt")

    def test_prefers_text_fence(self) -> None:
        md = """\
## My Prompt

```python
print("not this")
```

```text
The actual prompt content.
```
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert result == "The actual prompt content.\n"

    def test_supports_nested_fences(self) -> None:
        md = """\
## My Prompt

````text
Here is an inner fence:
```python
print("nested")
```
End of prompt.
````
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert '```python' in result
        assert 'print("nested")' in result
        assert "End of prompt." in result

    def test_prose_stops_at_next_heading(self) -> None:
        md = """\
## First

No fence here.

## Second

```text
Prompt in second section.
```
"""
        result = extract_prompt_from_markdown(md, "First")
        assert result == "No fence here.\n"
        assert "Prompt in second section" not in result

    def test_falls_back_to_first_fence_if_no_text(self) -> None:
        md = """\
## My Prompt

```yaml
key: value
```
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert result == "key: value\n"


class TestSanitizeDirname:
    def test_normal_name(self) -> None:
        assert sanitize_dirname("my-plan") == "my-plan"

    def test_special_chars_replaced(self) -> None:
        result = sanitize_dirname("my plan!@#2024")
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result
        assert " " not in result
        assert "_" in result

    def test_empty_returns_unnamed(self) -> None:
        assert sanitize_dirname("") == "unnamed"

    def test_only_special_chars_returns_unnamed(self) -> None:
        assert sanitize_dirname("!!!") == "unnamed"

    def test_underscores_and_hyphens_preserved(self) -> None:
        assert sanitize_dirname("my_plan-v2") == "my_plan-v2"

    def test_dots_replaced(self) -> None:
        result = sanitize_dirname("my.plan")
        assert "." not in result


class TestRenderTemplate:
    def test_basic_substitution(self) -> None:
        result = render_template("Hello {{ name }}", {"name": "world"})
        assert result == "Hello world"

    def test_unknown_vars_left_as_is(self) -> None:
        result = render_template("{{ known }} and {{ unknown }}", {"known": "yes"})
        assert result == "yes and {{ unknown }}"

    def test_dotted_names(self) -> None:
        result = render_template(
            "Status: {{ task-a.status }}, Code: {{ task-a.exit_code }}",
            {"task-a.status": "success", "task-a.exit_code": "0"},
        )
        assert result == "Status: success, Code: 0"

    def test_multiple_vars_in_one_string(self) -> None:
        result = render_template(
            "{{ a }}/{{ b }}/{{ c }}",
            {"a": "x", "b": "y", "c": "z"},
        )
        assert result == "x/y/z"

    def test_empty_variables_dict(self) -> None:
        text = "{{ workspace_root }}/src"
        result = render_template(text, {})
        assert result == "{{ workspace_root }}/src"

    def test_no_placeholders(self) -> None:
        result = render_template("plain text", {"key": "val"})
        assert result == "plain text"

    def test_whitespace_around_var_name(self) -> None:
        result = render_template("{{name}}", {"name": "val"})
        assert result == "val"

    def test_extra_whitespace_around_var_name(self) -> None:
        result = render_template("{{  name  }}", {"name": "val"})
        assert result == "val"


class TestHumanizeOutputLine:
    """Tests for humanize_output_line() — codex JSON → readable text."""

    def test_plain_text_unchanged(self) -> None:
        assert humanize_output_line("hello world") == "hello world"

    def test_non_json_brace_unchanged(self) -> None:
        assert humanize_output_line("{not valid json") == "{not valid json"

    def test_agent_message_extracts_text(self) -> None:
        import json
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Vou analisar os controladores existentes\nSegunda linha"},
        })
        result = humanize_output_line(line)
        assert result == "Vou analisar os controladores existentes"

    def test_agent_message_strips_markdown_bold(self) -> None:
        import json
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "**Rerunning search patterns**"},
        })
        assert humanize_output_line(line) == "Rerunning search patterns"

    def test_command_execution_started(self) -> None:
        import json
        line = json.dumps({
            "type": "item.started",
            "item": {"type": "command_execution", "command": "\"C:\\\\Program Files\\\\PowerShell\\\\7\\\\pwsh.exe\" -Command \"Get-Content -Path foo\""},
        })
        result = humanize_output_line(line)
        assert result.startswith("$ pwsh.exe")

    def test_command_execution_completed(self) -> None:
        import json
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "git status"},
        })
        result = humanize_output_line(line)
        assert result == "cmd done: git status"

    def test_reasoning_event(self) -> None:
        import json
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "reasoning", "text": "I need to think about this carefully"},
        })
        assert humanize_output_line(line) == "thinking: I need to think about this carefully"

    def test_response_completed_suppressed(self) -> None:
        import json
        line = json.dumps({"type": "response.completed"})
        assert humanize_output_line(line) == ""

    def test_turn_completed_suppressed(self) -> None:
        import json
        line = json.dumps({"type": "turn.completed"})
        assert humanize_output_line(line) == ""

    def test_rate_limit_event_suppressed(self) -> None:
        import json
        line = json.dumps({"type": "rate_limit_event"})
        assert humanize_output_line(line) == ""

    def test_unknown_type_shows_label(self) -> None:
        import json
        line = json.dumps({"type": "rate_limits.updated"})
        assert humanize_output_line(line) == "rate limits updated"

    def test_truncates_long_text_default(self) -> None:
        import json
        long_text = "A" * 300
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": long_text},
        })
        result = humanize_output_line(line)
        assert len(result) == 200
        assert result.endswith("\u2026")

    def test_truncates_long_text_custom_max(self) -> None:
        import json
        long_text = "A" * 300
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": long_text},
        })
        result = humanize_output_line(line, max_len=120)
        assert len(result) == 120
        assert result.endswith("\u2026")

    def test_empty_line_unchanged(self) -> None:
        assert humanize_output_line("") == ""

    def test_empty_json_object_unchanged(self) -> None:
        assert humanize_output_line("{}") == "{}"


# ---------------------------------------------------------------------------
# Additional tests (appended)
# ---------------------------------------------------------------------------

import json

from maestro_cli.utils import (
    evaluate_when_condition,
    extract_structured_context,
    format_cost,
    format_duration,
    format_structured_context,
    build_summarization_prompt,
    build_reduce_prompt,
    _truncate,
)
from maestro_cli.models import StructuredContext


class TestFormatDuration:
    """Tests for format_duration() helper."""

    def test_none_returns_dash(self) -> None:
        assert format_duration(None) == "--"

    def test_zero(self) -> None:
        assert format_duration(0.0) == "0.0s"

    def test_sub_second(self) -> None:
        assert format_duration(0.3) == "0.3s"

    def test_seconds(self) -> None:
        assert format_duration(45.7) == "45.7s"

    def test_exactly_sixty(self) -> None:
        assert format_duration(60.0) == "1m00s"

    def test_minutes_and_seconds(self) -> None:
        assert format_duration(125.0) == "2m05s"

    def test_just_under_hour(self) -> None:
        assert format_duration(3599.0) == "59m59s"

    def test_exactly_one_hour(self) -> None:
        assert format_duration(3600.0) == "1h00m"

    def test_hours_and_minutes(self) -> None:
        assert format_duration(7320.0) == "2h02m"

    def test_large_duration(self) -> None:
        assert format_duration(86400.0) == "24h00m"

    def test_negative_handled(self) -> None:
        # Negative values — format_duration doesn't reject them
        result = format_duration(-5.0)
        assert isinstance(result, str)


class TestFormatCost:
    """Tests for format_cost() helper."""

    def test_none_returns_dash(self) -> None:
        assert format_cost(None) == "--"

    def test_zero(self) -> None:
        assert format_cost(0.0) == "$0.00"

    def test_small_cost(self) -> None:
        assert format_cost(0.001) == "$0.00"

    def test_normal_cost(self) -> None:
        assert format_cost(1.23) == "$1.23"

    def test_large_cost(self) -> None:
        assert format_cost(999.99) == "$999.99"

    def test_rounds_down(self) -> None:
        assert format_cost(0.004) == "$0.00"

    def test_rounds_up(self) -> None:
        assert format_cost(0.005) == "$0.01"

    def test_integer_input(self) -> None:
        assert format_cost(5) == "$5.00"


class TestTruncate:
    """Tests for _truncate() helper."""

    def test_short_text_unchanged(self) -> None:
        assert _truncate("hello") == "hello"

    def test_exact_length_unchanged(self) -> None:
        text = "x" * 200
        assert _truncate(text) == text

    def test_over_length_truncated(self) -> None:
        text = "A" * 201
        result = _truncate(text)
        assert len(result) == 200
        assert result.endswith("\u2026")

    def test_custom_max_len(self) -> None:
        result = _truncate("hello world", max_len=5)
        assert len(result) == 5
        assert result.endswith("\u2026")

    def test_empty_string(self) -> None:
        assert _truncate("") == ""


class TestResolvePathExtended:
    """Extended tests for resolve_path()."""

    def test_resolve_to_absolute(self, tmp_path: Path) -> None:
        result = resolve_path(tmp_path, "a/b/c.txt")
        assert result is not None
        assert result.is_absolute()

    def test_parent_traversal(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        result = resolve_path(sub, "../sibling.txt")
        assert result is not None
        assert result.is_absolute()
        assert result.parent == tmp_path

    def test_dot_relative(self, tmp_path: Path) -> None:
        result = resolve_path(tmp_path, "./file.txt")
        assert result is not None
        assert result.name == "file.txt"
        assert str(tmp_path) in str(result)

    def test_forward_slashes_on_windows(self, tmp_path: Path) -> None:
        result = resolve_path(tmp_path, "sub/dir/file.txt")
        assert result is not None
        assert result.name == "file.txt"


class TestRenderTemplateExtended:
    """Extended tests for render_template()."""

    def test_adjacent_vars(self) -> None:
        result = render_template("{{ a }}{{ b }}", {"a": "x", "b": "y"})
        assert result == "xy"

    def test_var_at_start(self) -> None:
        result = render_template("{{ x }} end", {"x": "start"})
        assert result == "start end"

    def test_var_at_end(self) -> None:
        result = render_template("begin {{ x }}", {"x": "end"})
        assert result == "begin end"

    def test_same_var_multiple_times(self) -> None:
        result = render_template("{{ x }}/{{ x }}", {"x": "val"})
        assert result == "val/val"

    def test_empty_value(self) -> None:
        result = render_template("prefix {{ x }} suffix", {"x": ""})
        assert result == "prefix  suffix"

    def test_value_with_braces(self) -> None:
        result = render_template("{{ x }}", {"x": "{ not a var }"})
        assert result == "{ not a var }"

    def test_real_global_vars(self) -> None:
        result = render_template(
            "{{ workspace_root }}/{{ plan_name }}/{{ task_id }}",
            {"workspace_root": "/code", "plan_name": "my-plan", "task_id": "t1"},
        )
        assert result == "/code/my-plan/t1"

    def test_dotted_output_var(self) -> None:
        result = render_template(
            "{{ my-task.output.field_name }}",
            {"my-task.output.field_name": "some_value"},
        )
        assert result == "some_value"

    def test_mixed_known_unknown(self) -> None:
        result = render_template(
            "{{ known }}/{{ unknown }}/{{ known }}",
            {"known": "yes"},
        )
        assert result == "yes/{{ unknown }}/yes"

    def test_literal_braces_not_affected(self) -> None:
        # A single brace is not a template
        result = render_template("{ not_a_var }", {"not_a_var": "bad"})
        assert result == "{ not_a_var }"

    def test_triple_braces_partial_match(self) -> None:
        # {{{ x }}} — the inner {{ x }} should match
        result = render_template("{{{ x }}}", {"x": "val"})
        assert "val" in result


class TestExtractPromptFromMarkdownExtended:
    """Extended tests for extract_prompt_from_markdown()."""

    def test_h1_heading_stops_section(self) -> None:
        md = """\
## My Prompt

Content here.

# Top Level
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert result == "Content here.\n"
        assert "Top Level" not in result

    def test_multiple_fences_prefers_text(self) -> None:
        md = """\
## My Prompt

```python
code here
```

```yaml
key: value
```

```text
actual prompt
```
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert result == "actual prompt\n"

    def test_heading_with_extra_whitespace_not_matched(self) -> None:
        md = """\
##   My Prompt

```text
content
```
"""
        # Extra whitespace between ## and heading text means no match
        with pytest.raises(ValueError, match="Heading not found"):
            extract_prompt_from_markdown(md, "My Prompt")

    def test_heading_exact_whitespace_match(self) -> None:
        md = """\
## My Prompt

```text
content
```
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert result == "content\n"

    def test_last_section_no_next_heading(self) -> None:
        md = """\
## First

Some text.

## Last

Final section with no following heading.
"""
        result = extract_prompt_from_markdown(md, "Last")
        assert "Final section" in result

    def test_multiline_prose_preserves_blank_lines(self) -> None:
        md = """\
## My Prompt

Line one.

Line three after blank.
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert "Line one." in result
        assert "Line three after blank." in result

    def test_fence_with_backtick_count_four(self) -> None:
        md = """\
## My Prompt

````text
content with triple backticks:
```
nested
```
done
````
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert "content with triple backticks:" in result
        assert "nested" in result
        assert "done" in result

    def test_non_text_fence_fallback(self) -> None:
        md = """\
## My Prompt

```json
{"key": "value"}
```
"""
        result = extract_prompt_from_markdown(md, "My Prompt")
        assert '{"key": "value"}' in result


class TestSanitizeDirnameExtended:
    """Extended tests for sanitize_dirname()."""

    def test_unicode_replaced(self) -> None:
        result = sanitize_dirname("café")
        assert "é" not in result

    def test_leading_trailing_special_stripped(self) -> None:
        result = sanitize_dirname("__name__")
        assert result == "name"

    def test_numeric_name(self) -> None:
        assert sanitize_dirname("12345") == "12345"

    def test_single_valid_char(self) -> None:
        assert sanitize_dirname("a") == "a"

    def test_slashes_replaced(self) -> None:
        result = sanitize_dirname("path/to/dir")
        assert "/" not in result
        assert "\\" not in result


class TestEvaluateWhenCondition:
    """Tests for evaluate_when_condition()."""

    def test_equality_true(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ task-a.status }} == success",
            {"task-a.status": "success"},
        )
        assert result is True

    def test_equality_false(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ task-a.status }} == success",
            {"task-a.status": "failed"},
        )
        assert result is False

    def test_inequality_true(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ task-a.status }} != failed",
            {"task-a.status": "success"},
        )
        assert result is True

    def test_inequality_false(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ task-a.status }} != success",
            {"task-a.status": "success"},
        )
        assert result is False

    def test_invalid_expression_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid when expression"):
            evaluate_when_condition("just text no operator", {})

    def test_unresolved_var_left_as_is(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ unknown }} == value",
            {},
        )
        # {{ unknown }} stays as literal, so "{{ unknown }}" != "value"
        assert result is False
        assert "{{ unknown }}" in rendered

    def test_rendered_output_contains_resolved_values(self) -> None:
        _, rendered = evaluate_when_condition(
            "{{ x }} == yes",
            {"x": "yes"},
        )
        assert "yes == yes" in rendered


class TestExtractStructuredContext:
    """Tests for extract_structured_context()."""

    def test_empty_log_file(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert ctx.task_id == "t1"
        assert ctx.status == "success"
        assert ctx.files_changed == []
        assert ctx.errors == []
        assert ctx.warnings == []

    def test_missing_log_file(self, tmp_path: Path) -> None:
        log = tmp_path / "nonexistent.log"
        ctx = extract_structured_context(log, "t1", "failed", 1, 2.0, 0.5)
        assert ctx.task_id == "t1"
        assert ctx.status == "failed"
        assert ctx.cost_usd == 0.5

    def test_git_status_files(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("M  src/main.py\nA  src/new.py\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert "src/main.py" in ctx.files_changed
        assert "src/new.py" in ctx.files_changed

    def test_git_diff_files(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("+++ b/src/changed.py\n--- a/src/old.py\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert "src/changed.py" in ctx.files_changed
        assert "src/old.py" in ctx.files_changed

    def test_deduplicates_files(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("M  src/main.py\nM  src/main.py\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert ctx.files_changed.count("src/main.py") == 1

    def test_error_lines_from_stderr(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("[stderr] error: something broke\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "failed", 1, 1.0, None)
        assert any("something broke" in e for e in ctx.errors)

    def test_error_lines_from_traceback(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("Traceback (most recent call last):\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "failed", 1, 1.0, None)
        assert len(ctx.errors) == 1

    def test_warning_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("[stderr] warning: deprecated usage\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert any("deprecated" in w for w in ctx.warnings)

    def test_json_result_extraction(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        data = json.dumps({"type": "result", "result": "Task completed successfully"})
        log.write_text(data + "\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert ctx.result_text == "Task completed successfully"
        assert len(ctx.decisions) == 1

    def test_cost_usd_passed_through(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        log.write_text("ok\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, 3.14)
        assert ctx.cost_usd == 3.14

    def test_caps_files_at_max(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        lines = [f"M  src/file{i}.py" for i in range(150)]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "success", 0, 1.0, None)
        assert len(ctx.files_changed) <= 100

    def test_caps_errors_at_max(self, tmp_path: Path) -> None:
        log = tmp_path / "task.log"
        lines = [f"[stderr] error: problem {i}" for i in range(60)]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ctx = extract_structured_context(log, "t1", "failed", 1, 1.0, None)
        assert len(ctx.errors) <= 50


class TestFormatStructuredContext:
    """Tests for format_structured_context()."""

    def test_minimal_context(self) -> None:
        ctx = StructuredContext(task_id="t1", status="success", exit_code=0, duration_sec=1.0)
        text = format_structured_context(ctx)
        assert "t1" in text
        assert "success" in text
        assert "1.0s" in text

    def test_context_with_files(self) -> None:
        ctx = StructuredContext(
            task_id="t1", status="success", exit_code=0, duration_sec=1.0,
            files_changed=["a.py", "b.py"],
        )
        text = format_structured_context(ctx)
        assert "Files changed (2)" in text
        assert "a.py" in text
        assert "b.py" in text

    def test_context_with_errors(self) -> None:
        ctx = StructuredContext(
            task_id="t1", status="failed", exit_code=1, duration_sec=2.0,
            errors=["TypeError: bad argument"],
        )
        text = format_structured_context(ctx)
        assert "Errors (1)" in text
        assert "TypeError" in text

    def test_context_with_warnings(self) -> None:
        ctx = StructuredContext(
            task_id="t1", status="success", exit_code=0, duration_sec=1.0,
            warnings=["DeprecationWarning: old API"],
        )
        text = format_structured_context(ctx)
        assert "Warnings (1)" in text
        assert "DeprecationWarning" in text

    def test_context_with_decisions(self) -> None:
        ctx = StructuredContext(
            task_id="t1", status="success", exit_code=0, duration_sec=1.0,
            decisions=["Refactored module X"],
        )
        text = format_structured_context(ctx)
        assert "Key outcomes" in text
        assert "Refactored module X" in text


class TestBuildSummarizationPrompt:
    """Tests for build_summarization_prompt()."""

    def test_basic_prompt(self) -> None:
        ctx = StructuredContext(task_id="t1", status="success", exit_code=0, duration_sec=1.0)
        prompt = build_summarization_prompt("t1", "output here\n", ctx)
        assert "t1" in prompt
        assert "success" in prompt
        assert "output here" in prompt
        assert "<analysis>" in prompt
        assert "Primary Request" in prompt
        assert "Next Steps" in prompt

    def test_prompt_with_files(self) -> None:
        ctx = StructuredContext(
            task_id="t1", status="success", exit_code=0, duration_sec=1.0,
            files_changed=["a.py", "b.py"],
        )
        prompt = build_summarization_prompt("t1", "", ctx)
        assert "Files changed (2)" in prompt
        assert "a.py" in prompt

    def test_prompt_with_errors(self) -> None:
        ctx = StructuredContext(
            task_id="t1", status="failed", exit_code=1, duration_sec=2.0,
            errors=["broken import"],
        )
        prompt = build_summarization_prompt("t1", "", ctx)
        assert "Errors (1)" in prompt
        assert "broken import" in prompt

    def test_prompt_truncates_many_files(self) -> None:
        ctx = StructuredContext(
            task_id="t1", status="success", exit_code=0, duration_sec=1.0,
            files_changed=[f"file{i}.py" for i in range(30)],
        )
        prompt = build_summarization_prompt("t1", "", ctx)
        assert "and 10 more" in prompt

    def test_empty_stdout_tail(self) -> None:
        ctx = StructuredContext(task_id="t1", status="success", exit_code=0, duration_sec=1.0)
        prompt = build_summarization_prompt("t1", "", ctx)
        assert "Task output" not in prompt

    def test_whitespace_stdout_tail(self) -> None:
        ctx = StructuredContext(task_id="t1", status="success", exit_code=0, duration_sec=1.0)
        prompt = build_summarization_prompt("t1", "   \n  \n", ctx)
        assert "Task output" not in prompt


class TestBuildReducePrompt:
    """Tests for build_reduce_prompt()."""

    def test_basic_reduce(self) -> None:
        summaries = {"t1": "Task 1 did X", "t2": "Task 2 did Y"}
        prompt = build_reduce_prompt(summaries)
        assert "Synthesize" in prompt
        assert "### t1" in prompt
        assert "### t2" in prompt
        assert "Task 1 did X" in prompt
        assert "Task 2 did Y" in prompt

    def test_empty_summaries(self) -> None:
        prompt = build_reduce_prompt({})
        assert "Synthesize" in prompt
        assert "Verdict" in prompt

    def test_single_summary(self) -> None:
        prompt = build_reduce_prompt({"only": "Single task summary"})
        assert "### only" in prompt
        assert "Single task summary" in prompt


class TestHumanizeOutputLineExtended:
    """Extended tests for humanize_output_line()."""

    def test_claude_result_event(self) -> None:
        line = json.dumps({"type": "result", "result": "All tests passed.\nDetails here."})
        result = humanize_output_line(line)
        assert result == "All tests passed."

    def test_claude_result_empty_result(self) -> None:
        line = json.dumps({"type": "result", "result": ""})
        result = humanize_output_line(line)
        assert result == ""

    def test_claude_system_event_suppressed(self) -> None:
        line = json.dumps({"type": "system", "data": "something"})
        assert humanize_output_line(line) == ""

    def test_claude_user_event_suppressed(self) -> None:
        line = json.dumps({"type": "user", "content": "question"})
        assert humanize_output_line(line) == ""

    def test_claude_assistant_text(self) -> None:
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Here is my analysis.\nMore details."}]},
        })
        result = humanize_output_line(line)
        assert result == "Here is my analysis."

    def test_claude_assistant_tool_use(self) -> None:
        line = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "read_file"}]},
        })
        result = humanize_output_line(line)
        assert result == "tool: read_file"

    def test_claude_assistant_empty_content(self) -> None:
        line = json.dumps({"type": "assistant", "message": {"content": []}})
        assert humanize_output_line(line) == ""

    def test_claude_assistant_no_message(self) -> None:
        line = json.dumps({"type": "assistant"})
        assert humanize_output_line(line) == ""

    def test_codex_item_started_no_command(self) -> None:
        line = json.dumps({
            "type": "item.started",
            "item": {"type": "command_execution", "command": ""},
        })
        result = humanize_output_line(line)
        assert result == "running command..."

    def test_codex_item_completed_no_command(self) -> None:
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "command_execution", "command": ""},
        })
        result = humanize_output_line(line)
        assert result == "command completed"

    def test_codex_agent_message_empty_text(self) -> None:
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": ""},
        })
        # Empty text falls through
        result = humanize_output_line(line)
        assert isinstance(result, str)

    def test_codex_reasoning_empty_text(self) -> None:
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "reasoning", "text": ""},
        })
        result = humanize_output_line(line)
        assert isinstance(result, str)

    def test_rate_limit_suppressed(self) -> None:
        line = json.dumps({"type": "rate.limit"})
        assert humanize_output_line(line) == ""

    def test_item_started_lifecycle_suppressed(self) -> None:
        line = json.dumps({"type": "item.started"})
        assert humanize_output_line(line) == ""

    def test_response_started_suppressed(self) -> None:
        line = json.dumps({"type": "response.started"})
        assert humanize_output_line(line) == ""

    def test_turn_started_suppressed(self) -> None:
        line = json.dumps({"type": "turn.started"})
        assert humanize_output_line(line) == ""

    def test_non_dict_json(self) -> None:
        line = json.dumps([1, 2, 3])
        assert humanize_output_line(line) == line

    def test_json_with_unknown_item_type(self) -> None:
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "unknown_thing"},
        })
        result = humanize_output_line(line)
        # Falls through to generic event type handler
        assert "item completed" in result

    def test_whitespace_preserved_in_non_json(self) -> None:
        assert humanize_output_line("  indented line  ") == "  indented line  "

    def test_json_string_not_dict(self) -> None:
        line = json.dumps("just a string")
        assert humanize_output_line(line) == line

    def test_command_with_complex_path(self) -> None:
        line = json.dumps({
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "/usr/local/bin/python3 -m pytest tests/"},
        })
        result = humanize_output_line(line)
        assert "python3" in result
        assert "cmd done:" in result
