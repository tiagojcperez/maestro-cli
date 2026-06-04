from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli import workspace_assertions
from maestro_cli.workspace_assertions import (
    evaluate_workspace_assertion,
    normalize_workspace_assertion,
)


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — count/min_count int() failure branch
# ---------------------------------------------------------------------------


class TestCountParsingErrors:
    def test_count_non_numeric_string_raises_valueerror(self) -> None:
        """A non-numeric ``count`` string fails int() (ValueError) -> remapped."""
        a = {
            "type": "file_contains_count",
            "path": "src/main.py",
            "pattern": "needle",
            "count": "not-a-number",
        }
        with pytest.raises(ValueError, match=r"count must be an integer >= 0"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_min_count_non_numeric_string_raises_valueerror(self) -> None:
        """A non-numeric ``min_count`` string fails int() -> remapped error."""
        a = {
            "type": "file_contains_count",
            "path": "src/main.py",
            "pattern": "needle",
            "min_count": "xyz",
        }
        with pytest.raises(ValueError, match=r"min_count must be an integer >= 0"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_count_unconvertible_type_raises_valueerror(self) -> None:
        """A list value triggers TypeError inside int() -> remapped error."""
        a = {
            "type": "file_contains_count",
            "path": "src/main.py",
            "pattern": "needle",
            "count": ["nope"],
        }
        with pytest.raises(ValueError, match=r"count must be an integer >= 0"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# _eval_file_content_assertion — OSError on read_text branch
# ---------------------------------------------------------------------------


class TestFileReadError:
    def test_read_text_oserror_reports_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When read_text raises OSError on an existing file, return a failure."""
        target = tmp_path / "main.py"
        target.write_text("def hello(): pass\n", encoding="utf-8")

        real_read_text = Path.read_text

        def boom(self: Path, *args: object, **kwargs: object) -> str:
            if self == target:
                raise OSError("simulated read failure")
            return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", boom)

        assertion = {
            "type": "file_contains",
            "path": "main.py",
            "pattern": "def hello",
        }
        passed, message = evaluate_workspace_assertion(assertion, tmp_path)

        assert passed is False
        assert "Failed to read file" in message
        assert "simulated read failure" in message
