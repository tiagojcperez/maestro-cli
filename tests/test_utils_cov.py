from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli import utils
from maestro_cli.models import StructuredContext
from maestro_cli.utils import (
    build_summarization_prompt,
    command_to_string,
    extract_structured_context,
    humanize_output_line,
)


class TestCommandToStringPosix:
    def test_posix_uses_shlex_join(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On POSIX, a list command is rendered via shlex.join.

        The function performs no pathlib operations, so swapping the os.name
        the module sees is safe here (no WindowsPath construction occurs).
        """
        monkeypatch.setattr(utils.os, "name", "posix")
        result = command_to_string(["echo", "hello world"])
        # shlex.join quotes args with spaces using single quotes.
        assert "echo" in result
        assert "'hello world'" in result


class TestExtractStructuredContextJsonError:
    def test_invalid_json_result_line_is_ignored(self, tmp_path: Path) -> None:
        """A line that looks like a JSON result but fails to parse is skipped.

        Drives the except (JSONDecodeError, ValueError): pass branch.
        """
        log = tmp_path / "task.log"
        # Contains the trigger substrings '"type"' and '"result"' so the JSON
        # parse is attempted, but the line is not valid JSON -> decode error.
        log.write_text('{"type": "result", "result": NOT_VALID_JSON}\n', encoding="utf-8")

        ctx = extract_structured_context(
            log_path=log,
            task_id="t1",
            status="success",
            exit_code=0,
            duration_sec=1.0,
            cost_usd=None,
        )

        # The malformed line yields no result_text (parse failed and was swallowed).
        assert ctx.result_text == ""
        assert ctx.task_id == "t1"

    def test_valid_json_result_line_populates_result_text(self, tmp_path: Path) -> None:
        """Sanity check the happy path so the except branch is the only diff."""
        log = tmp_path / "task.log"
        payload = json.dumps({"type": "result", "result": "all done"})
        log.write_text(payload + "\n", encoding="utf-8")

        ctx = extract_structured_context(
            log_path=log,
            task_id="t2",
            status="success",
            exit_code=0,
            duration_sec=1.0,
            cost_usd=None,
        )
        assert ctx.result_text == "all done"


class TestBuildSummarizationPromptSections:
    def _ctx(self, **kwargs: object) -> StructuredContext:
        base: dict[str, object] = {
            "task_id": "t",
            "status": "success",
            "exit_code": 0,
            "duration_sec": 2.0,
        }
        base.update(kwargs)
        return StructuredContext(**base)  # type: ignore[arg-type]

    def test_warnings_section_rendered(self) -> None:
        """Warnings list drives the warnings block."""
        ctx = self._ctx(warnings=["deprecated API used", "unused import"])
        prompt = build_summarization_prompt("t", "", ctx)
        assert "Warnings (2)" in prompt
        assert "deprecated API used" in prompt
        assert "unused import" in prompt

    def test_decisions_section_rendered(self) -> None:
        """Decisions list drives the decisions block."""
        ctx = self._ctx(decisions=["chose approach A", "skipped step B"])
        prompt = build_summarization_prompt("t", "", ctx)
        assert "Decisions (2)" in prompt
        assert "chose approach A" in prompt
        assert "skipped step B" in prompt

    def test_warnings_and_decisions_together(self) -> None:
        ctx = self._ctx(
            warnings=["w1"],
            decisions=["d1"],
            errors=["e1"],
            files_changed=["a.py"],
        )
        prompt = build_summarization_prompt("t", "tail output", ctx)
        assert "Warnings (1)" in prompt
        assert "Decisions (1)" in prompt
        assert "Errors (1)" in prompt
        assert "Files changed (1)" in prompt


class TestHumanizeOutputLineCommandQuoting:
    def test_unbalanced_quote_falls_back_to_split(self) -> None:
        """A command with an unbalanced quote makes shlex.split raise.

        Drives the except ValueError: parts = normalized.split() fallback.
        """
        line = json.dumps(
            {
                "type": "item.started",
                "item": {"type": "command_execution", "command": 'echo "unterminated'},
            }
        )
        result = humanize_output_line(line)
        # Fallback split keeps the first whitespace token ("echo") as the binary.
        assert "echo" in result
        assert result.startswith("$ ")

    def test_unbalanced_quote_completed_event(self) -> None:
        line = json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "rm 'oops"},
            }
        )
        result = humanize_output_line(line)
        assert "cmd done:" in result
        assert "rm" in result

    def test_well_formed_command_uses_shlex(self) -> None:
        """Sanity: a balanced command parses via shlex (no fallback)."""
        line = json.dumps(
            {
                "type": "item.started",
                "item": {
                    "type": "command_execution",
                    "command": "git status --porcelain",
                },
            }
        )
        result = humanize_output_line(line)
        assert result.startswith("$ ")
        assert "git" in result
