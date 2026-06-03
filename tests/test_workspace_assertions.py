from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.workspace_assertions import (
    describe_workspace_assertion,
    evaluate_workspace_assertion,
    normalize_workspace_assertion,
)


# ---------------------------------------------------------------------------
# normalize_workspace_assertion
# ---------------------------------------------------------------------------


class TestNormalizeWorkspaceAssertion:
    def test_valid_file_contains(self) -> None:
        a = {"type": "file_contains", "path": "src/main.py", "pattern": "def hello"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "file_contains"
        assert result["path"] == "src/main.py"
        assert result["pattern"] == "def hello"

    def test_valid_file_contains_count_exact(self) -> None:
        a = {
            "type": "file_contains_count",
            "path": "src/main.py",
            "pattern": "ReportPdfService",
            "count": 3,
        }
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "file_contains_count"
        assert result["count"] == 3

    def test_valid_file_contains_count_minimum(self) -> None:
        a = {
            "type": "file_contains_count",
            "path": "src/main.py",
            "pattern": "ReportPdfService",
            "min_count": 2,
        }
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "file_contains_count"
        assert result["min_count"] == 2

    def test_valid_file_not_contains(self) -> None:
        a = {"type": "file_not_contains", "path": "src/main.py", "pattern": "debug"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "file_not_contains"

    def test_valid_file_regex(self) -> None:
        a = {"type": "file_regex", "path": "src/main.py", "pattern": r"def \w+"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "file_regex"

    def test_valid_file_regex_absent(self) -> None:
        a = {"type": "file_regex_absent", "path": "src/main.py", "pattern": r"eval\("}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "file_regex_absent"

    def test_valid_glob_exists(self) -> None:
        a = {"type": "glob_exists", "glob": "**/*.py"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "glob_exists"
        assert result["glob"] == "**/*.py"

    def test_valid_json_path_exists(self) -> None:
        a = {"type": "json_path_exists", "path": "config.json", "json_path": "database.host"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "json_path_exists"
        assert result["json_path"] == "database.host"

    def test_valid_composer_package_present(self) -> None:
        a = {"type": "composer_package_present", "package": "vendor/pkg"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "composer_package_present"
        assert result["package"] == "vendor/pkg"

    def test_valid_npm_package_present(self) -> None:
        a = {"type": "npm_package_present", "package": "express"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["type"] == "npm_package_present"

    def test_unknown_type_raises_value_error(self) -> None:
        a = {"type": "nonexistent_type", "path": "x", "pattern": "y"}
        with pytest.raises(ValueError, match="not valid"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_missing_path_for_file_contains_raises(self) -> None:
        a = {"type": "file_contains", "pattern": "hello"}
        with pytest.raises(ValueError, match="path is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_missing_pattern_for_file_contains_raises(self) -> None:
        a = {"type": "file_contains", "path": "src/main.py"}
        with pytest.raises(ValueError, match="pattern is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_file_contains_count_requires_count_or_min_count(self) -> None:
        a = {
            "type": "file_contains_count",
            "path": "src/main.py",
            "pattern": "needle",
        }
        with pytest.raises(ValueError, match="requires exactly one of 'count' or 'min_count'"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_file_contains_count_rejects_both_count_and_min_count(self) -> None:
        a = {
            "type": "file_contains_count",
            "path": "src/main.py",
            "pattern": "needle",
            "count": 2,
            "min_count": 1,
        }
        with pytest.raises(ValueError, match="requires exactly one of 'count' or 'min_count'"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_file_contains_count_rejects_negative_count(self) -> None:
        a = {
            "type": "file_contains_count",
            "path": "src/main.py",
            "pattern": "needle",
            "count": -1,
        }
        with pytest.raises(ValueError, match="count must be an integer >= 0"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_missing_glob_for_glob_exists_raises(self) -> None:
        a = {"type": "glob_exists"}
        with pytest.raises(ValueError, match="glob is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_json_path_missing_path_raises(self) -> None:
        a = {"type": "json_path_exists", "json_path": "a.b"}
        with pytest.raises(ValueError, match="path is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_json_path_missing_json_path_raises(self) -> None:
        a = {"type": "json_path_exists", "path": "config.json"}
        with pytest.raises(ValueError, match="json_path is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_package_missing_for_composer_raises(self) -> None:
        a = {"type": "composer_package_present"}
        with pytest.raises(ValueError, match="package is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_package_missing_for_npm_raises(self) -> None:
        a = {"type": "npm_package_present"}
        with pytest.raises(ValueError, match="package is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_non_dict_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must be an object"):
            normalize_workspace_assertion("not a dict", "assert[0]")

    def test_invalid_severity_raises(self) -> None:
        a = {"type": "glob_exists", "glob": "*.py", "severity": "critical"}
        with pytest.raises(ValueError, match="severity"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_valid_severity_accepted(self) -> None:
        for sev in ("error", "warning", "info"):
            a = {"type": "glob_exists", "glob": "*.py", "severity": sev}
            result = normalize_workspace_assertion(a, "assert[0]")
            assert result["severity"] == sev

    def test_rule_field_preserved(self) -> None:
        a = {"type": "glob_exists", "glob": "*.py", "rule": "Python files present"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["rule"] == "Python files present"

    def test_id_field_aliased_to_rule(self) -> None:
        a = {"type": "glob_exists", "glob": "*.py", "id": "CHECK_001"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["rule"] == "CHECK_001"

    def test_whitespace_stripped_from_fields(self) -> None:
        a = {"type": "glob_exists", "glob": "  **/*.py  "}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["glob"] == "**/*.py"

    def test_empty_string_glob_raises(self) -> None:
        a = {"type": "glob_exists", "glob": "   "}
        with pytest.raises(ValueError, match="non-empty"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_task_id_field_preserved(self) -> None:
        a = {"type": "glob_exists", "glob": "*.py", "task_id": "build-step"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["task_id"] == "build-step"


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — file_contains / file_not_contains
# ---------------------------------------------------------------------------


class TestEvaluateFileContains:
    def test_content_present_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("def hello():\n    pass\n", encoding="utf-8")
        assertion = {"type": "file_contains", "path": str(f), "pattern": "def hello"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "found" in msg

    def test_content_absent_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("def goodbye():\n    pass\n", encoding="utf-8")
        assertion = {"type": "file_contains", "path": str(f), "pattern": "def hello"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not found" in msg

    def test_file_not_found_fails(self, tmp_path: Path) -> None:
        assertion = {
            "type": "file_contains",
            "path": str(tmp_path / "missing.py"),
            "pattern": "hello",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not found" in msg.lower() or "File not found" in msg

    def test_file_not_contains_passes_when_absent(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("def goodbye():\n    pass\n", encoding="utf-8")
        assertion = {"type": "file_not_contains", "path": str(f), "pattern": "def hello"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_file_not_contains_fails_when_present(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("def hello():\n    pass\n", encoding="utf-8")
        assertion = {"type": "file_not_contains", "path": str(f), "pattern": "def hello"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_relative_path_resolved_from_base_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("import os\n", encoding="utf-8")
        assertion = {"type": "file_contains", "path": "main.py", "pattern": "import os"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_file_contains_count_exact_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "deps.php"
        f.write_text("ReportPdfService\nfoo\nReportPdfService\nbar\nReportPdfService\n", encoding="utf-8")
        assertion = {
            "type": "file_contains_count",
            "path": str(f),
            "pattern": "ReportPdfService",
            "count": 3,
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "expected exactly 3" in msg

    def test_file_contains_count_exact_fails_on_mismatch(self, tmp_path: Path) -> None:
        f = tmp_path / "deps.php"
        f.write_text("ReportPdfService\nReportPdfService\n", encoding="utf-8")
        assertion = {
            "type": "file_contains_count",
            "path": str(f),
            "pattern": "ReportPdfService",
            "count": 3,
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "occurred 2 time(s)" in msg

    def test_file_contains_count_minimum_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "deps.php"
        f.write_text("a\nReportPdfService\nb\nReportPdfService\nc\n", encoding="utf-8")
        assertion = {
            "type": "file_contains_count",
            "path": str(f),
            "pattern": "ReportPdfService",
            "min_count": 2,
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "expected at least 2" in msg

    def test_file_contains_count_minimum_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "deps.php"
        f.write_text("a\nReportPdfService\n", encoding="utf-8")
        assertion = {
            "type": "file_contains_count",
            "path": str(f),
            "pattern": "ReportPdfService",
            "min_count": 2,
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "occurred 1 time(s)" in msg


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — file_regex / file_regex_absent
# ---------------------------------------------------------------------------


class TestEvaluateFileRegex:
    def test_regex_matches_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("version = '1.2.3'\n", encoding="utf-8")
        assertion = {
            "type": "file_regex",
            "path": str(f),
            "pattern": r"version\s*=\s*'[\d.]+'",
        }
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_regex_no_match_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("something else\n", encoding="utf-8")
        assertion = {"type": "file_regex", "path": str(f), "pattern": r"version\s*="}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_invalid_regex_returns_false_with_message(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("content\n", encoding="utf-8")
        assertion = {"type": "file_regex", "path": str(f), "pattern": "[invalid("}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "regex" in msg.lower() or "Invalid" in msg

    def test_file_regex_absent_passes_when_no_match(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("safe content\n", encoding="utf-8")
        assertion = {"type": "file_regex_absent", "path": str(f), "pattern": r"eval\("}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_file_regex_absent_fails_when_matches(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("eval(user_input)\n", encoding="utf-8")
        assertion = {"type": "file_regex_absent", "path": str(f), "pattern": r"eval\("}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_multiline_regex_works(self, tmp_path: Path) -> None:
        f = tmp_path / "main.py"
        f.write_text("class Foo:\n    pass\n", encoding="utf-8")
        assertion = {"type": "file_regex", "path": str(f), "pattern": r"^class \w+:"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — glob_exists
# ---------------------------------------------------------------------------


class TestEvaluateGlobExists:
    def test_glob_absolute_matches_file(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("x", encoding="utf-8")
        assertion = {"type": "glob_exists", "glob": str(tmp_path / "*.py")}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "matched" in msg

    def test_glob_absolute_no_match_fails(self, tmp_path: Path) -> None:
        assertion = {"type": "glob_exists", "glob": str(tmp_path / "*.rs")}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_glob_relative_uses_base_dir(self, tmp_path: Path) -> None:
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("x", encoding="utf-8")
        assertion = {"type": "glob_exists", "glob": "src/*.py"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_glob_relative_no_match(self, tmp_path: Path) -> None:
        assertion = {"type": "glob_exists", "glob": "src/*.py"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_recursive_glob_matches_nested_files(self, tmp_path: Path) -> None:
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        (sub / "deep.py").write_text("x", encoding="utf-8")
        assertion = {"type": "glob_exists", "glob": "**/*.py"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "matched" in msg


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — json_path_exists
# ---------------------------------------------------------------------------


class TestEvaluateJsonPathExists:
    def test_simple_key_found(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"host": "localhost"}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "host"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_nested_dot_path_found(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"db": {"host": "localhost"}}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "db.host"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_array_index_found(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"items": ["a", "b", "c"]}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "items[1]"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_missing_key_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"host": "localhost"}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "port"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_file_not_found_fails(self, tmp_path: Path) -> None:
        assertion = {
            "type": "json_path_exists",
            "path": str(tmp_path / "missing.json"),
            "json_path": "key",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not found" in msg.lower() or "JSON file not found" in msg

    def test_invalid_json_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("not json", encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "key"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_deeply_nested_path(self, tmp_path: Path) -> None:
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"a": {"b": {"c": "value"}}}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "a.b.c"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_array_index_out_of_bounds_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"items": ["a", "b"]}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "items[5]"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_dot_path_on_non_dict_fails(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"name": "hello"}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "name.sub"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_null_value_at_path_returns_false(self, tmp_path: Path) -> None:
        """A JSON null at the path is indistinguishable from missing → False."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"optional": None}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "optional"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — composer_package_present
# ---------------------------------------------------------------------------


class TestEvaluateComposerPackage:
    def test_package_in_require(self, tmp_path: Path) -> None:
        composer = tmp_path / "composer.json"
        composer.write_text(
            json.dumps({"require": {"vendor/pkg": "^1.0"}}), encoding="utf-8"
        )
        assertion = {"type": "composer_package_present", "package": "vendor/pkg"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_package_in_require_dev(self, tmp_path: Path) -> None:
        composer = tmp_path / "composer.json"
        composer.write_text(
            json.dumps({"require-dev": {"phpunit/phpunit": "^10.0"}}), encoding="utf-8"
        )
        assertion = {"type": "composer_package_present", "package": "phpunit/phpunit"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_package_missing(self, tmp_path: Path) -> None:
        composer = tmp_path / "composer.json"
        composer.write_text(
            json.dumps({"require": {"other/pkg": "^1.0"}}), encoding="utf-8"
        )
        assertion = {"type": "composer_package_present", "package": "vendor/pkg"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_composer_json_not_found(self, tmp_path: Path) -> None:
        assertion = {"type": "composer_package_present", "package": "vendor/pkg"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "composer.json" in msg.lower() or "not found" in msg.lower()

    def test_custom_composer_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "backend" / "composer.json"
        custom.parent.mkdir()
        custom.write_text(
            json.dumps({"require": {"vendor/pkg": "^1.0"}}), encoding="utf-8"
        )
        assertion = {
            "type": "composer_package_present",
            "path": "backend/composer.json",
            "package": "vendor/pkg",
        }
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — npm_package_present
# ---------------------------------------------------------------------------


class TestEvaluateNpmPackage:
    def test_package_in_dependencies(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({"dependencies": {"express": "^4.0"}}), encoding="utf-8")
        assertion = {"type": "npm_package_present", "package": "express"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_package_in_dev_dependencies(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({"devDependencies": {"jest": "^29.0"}}), encoding="utf-8")
        assertion = {"type": "npm_package_present", "package": "jest"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_package_in_peer_dependencies(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({"peerDependencies": {"react": "^18.0"}}), encoding="utf-8")
        assertion = {"type": "npm_package_present", "package": "react"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_package_missing(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({"dependencies": {"express": "^4.0"}}), encoding="utf-8")
        assertion = {"type": "npm_package_present", "package": "nonexistent"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_package_json_not_found(self, tmp_path: Path) -> None:
        assertion = {"type": "npm_package_present", "package": "express"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "package.json" in msg.lower() or "not found" in msg.lower()

    def test_package_in_optional_dependencies(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps({"optionalDependencies": {"fsevents": "^2.3"}}),
            encoding="utf-8",
        )
        assertion = {"type": "npm_package_present", "package": "fsevents"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_npm_invalid_json_fails(self, tmp_path: Path) -> None:
        pkg = tmp_path / "package.json"
        pkg.write_text("not valid json {{", encoding="utf-8")
        assertion = {"type": "npm_package_present", "package": "express"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "load" in msg.lower() or "json" in msg.lower() or "failed" in msg.lower()


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — file_contains with directory path
# ---------------------------------------------------------------------------


class TestEvaluateDirectoryPath:
    def test_directory_path_fails_for_file_contains(self, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        assertion = {"type": "file_contains", "path": str(sub), "pattern": "hello"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not a file" in msg.lower()


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — unsupported type
# ---------------------------------------------------------------------------


class TestEvaluateUnsupportedType:
    def test_unknown_type_returns_false(self, tmp_path: Path) -> None:
        assertion = {"type": "unknown_type"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "Unsupported" in msg or "unsupported" in msg.lower()


# ---------------------------------------------------------------------------
# describe_workspace_assertion
# ---------------------------------------------------------------------------


class TestDescribeWorkspaceAssertion:
    def test_returns_rule_when_present(self) -> None:
        assertion = {"type": "glob_exists", "glob": "*.py", "rule": "Python files exist"}
        desc = describe_workspace_assertion(assertion, 0)
        assert desc == "Python files exist"

    def test_returns_type_with_index_when_no_rule(self) -> None:
        assertion = {"type": "glob_exists", "glob": "*.py"}
        desc = describe_workspace_assertion(assertion, 3)
        assert desc == "glob_exists[3]"

    def test_returns_assert_fallback_when_no_type(self) -> None:
        assertion: dict = {}
        desc = describe_workspace_assertion(assertion, 5)
        assert desc == "assert[5]"

    def test_empty_rule_falls_back_to_type(self) -> None:
        assertion = {"type": "glob_exists", "glob": "*.py", "rule": "   "}
        desc = describe_workspace_assertion(assertion, 1)
        assert desc == "glob_exists[1]"

    def test_index_reflected_in_output(self) -> None:
        assertion = {"type": "file_contains", "path": "x", "pattern": "y"}
        assert describe_workspace_assertion(assertion, 0) == "file_contains[0]"
        assert describe_workspace_assertion(assertion, 99) == "file_contains[99]"


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — additional edge cases
# ---------------------------------------------------------------------------


class TestNormalizeWorkspaceAssertionAdditional:
    def test_message_field_preserved(self) -> None:
        a = {"type": "glob_exists", "glob": "**/*.py", "message": "No Python files found"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["message"] == "No Python files found"

    def test_empty_task_id_raises(self) -> None:
        a = {"type": "glob_exists", "glob": "*.py", "task_id": "   "}
        with pytest.raises(ValueError, match="task_id"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — additional package manager edge cases
# ---------------------------------------------------------------------------


class TestEvaluateComposerPackageAdditional:
    def test_invalid_composer_json_fails(self, tmp_path: Path) -> None:
        composer = tmp_path / "composer.json"
        composer.write_text("this is not valid json {{", encoding="utf-8")
        assertion = {"type": "composer_package_present", "package": "vendor/pkg"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "load" in msg.lower() or "json" in msg.lower() or "failed" in msg.lower()


class TestNormalizeWorkspaceAssertionEmptyMessage:
    def test_empty_message_field_raises(self) -> None:
        """A 'message' key whose value is blank must raise ValueError because
        normalize iterates over the key list and strips before checking."""
        a = {"type": "glob_exists", "glob": "*.py", "message": "   "}
        with pytest.raises(ValueError, match="non-empty"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_empty_severity_raises_non_empty(self) -> None:
        """An empty severity string should raise ValueError before reaching the
        validity check."""
        a = {"type": "glob_exists", "glob": "*.py", "severity": "   "}
        with pytest.raises(ValueError, match="non-empty"):
            normalize_workspace_assertion(a, "assert[0]")


class TestEvaluateNpmPackageAdditional:
    def test_custom_package_json_path(self, tmp_path: Path) -> None:
        frontend = tmp_path / "frontend"
        frontend.mkdir()
        (frontend / "package.json").write_text(
            json.dumps({"dependencies": {"react": "^18.0"}}), encoding="utf-8"
        )
        assertion = {
            "type": "npm_package_present",
            "path": "frontend/package.json",
            "package": "react",
        }
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — json_path_exists: falsy-but-not-None values
# ---------------------------------------------------------------------------


class TestEvaluateJsonPathFalsyValues:
    def test_false_value_at_path_returns_true(self, tmp_path: Path) -> None:
        """JSON boolean false is not None, so json_path_exists should pass."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"enabled": False}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "enabled"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_zero_value_at_path_returns_true(self, tmp_path: Path) -> None:
        """JSON number 0 is not None, so json_path_exists should pass."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"count": 0}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "count"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_empty_string_value_at_path_returns_true(self, tmp_path: Path) -> None:
        """JSON empty string is not None, so json_path_exists should pass."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"label": ""}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "label"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — file_regex_absent requires path and pattern
# ---------------------------------------------------------------------------


class TestNormalizeFileRegexAbsentRequirements:
    def test_file_regex_absent_missing_path_raises(self) -> None:
        a = {"type": "file_regex_absent", "pattern": r"eval\("}
        with pytest.raises(ValueError, match="path is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_file_regex_absent_missing_pattern_raises(self) -> None:
        a = {"type": "file_regex_absent", "path": "src/main.py"}
        with pytest.raises(ValueError, match="pattern is required"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — file_regex / file_regex_absent missing file
# ---------------------------------------------------------------------------


class TestEvaluateFileRegexMissingFile:
    def test_file_regex_file_not_found_returns_false(self, tmp_path: Path) -> None:
        """file_regex on a missing file should return False (hits the
        path.exists() guard in _eval_file_content_assertion)."""
        assertion = {
            "type": "file_regex",
            "path": str(tmp_path / "nonexistent.py"),
            "pattern": r"def \w+",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not found" in msg.lower()

    def test_file_regex_absent_file_not_found_returns_false(self, tmp_path: Path) -> None:
        """file_regex_absent on a missing file should also return False — the
        file cannot be asserted 'absent' of a pattern if it doesn't exist."""
        assertion = {
            "type": "file_regex_absent",
            "path": str(tmp_path / "nonexistent.py"),
            "pattern": r"eval\(",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not found" in msg.lower()


# ---------------------------------------------------------------------------
# _lookup_json_path — array index into a non-list value
# ---------------------------------------------------------------------------


class TestEvaluateFileContentMessages:
    def test_file_regex_success_message_contains_matched(self, tmp_path: Path) -> None:
        """file_regex on a matching file should include 'matched' in its message."""
        f = tmp_path / "src.py"
        f.write_text("def main(): pass\n", encoding="utf-8")
        assertion = {"type": "file_regex", "path": str(f), "pattern": r"def \w+"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "matched" in msg

    def test_file_not_contains_success_message_contains_not_found(
        self, tmp_path: Path
    ) -> None:
        """file_not_contains pass → message says 'not found'."""
        f = tmp_path / "app.py"
        f.write_text("safe content\n", encoding="utf-8")
        assertion = {"type": "file_not_contains", "path": str(f), "pattern": "eval("}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "not found" in msg

    def test_file_not_contains_failure_message_contains_found(self, tmp_path: Path) -> None:
        """file_not_contains fail → message says 'found'."""
        f = tmp_path / "app.py"
        f.write_text("eval(user_input)\n", encoding="utf-8")
        assertion = {"type": "file_not_contains", "path": str(f), "pattern": "eval("}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "found" in msg


class TestEvaluateComposerAndNpmMessages:
    def test_composer_success_message_says_is_present(self, tmp_path: Path) -> None:
        """On success, composer message must say 'is present'."""
        composer = tmp_path / "composer.json"
        composer.write_text(
            json.dumps({"require": {"vendor/pkg": "^1.0"}}), encoding="utf-8"
        )
        assertion = {"type": "composer_package_present", "package": "vendor/pkg"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "is present" in msg

    def test_composer_failure_message_says_is_missing(self, tmp_path: Path) -> None:
        """On failure, composer message must say 'is missing'."""
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({"require": {}}), encoding="utf-8")
        assertion = {"type": "composer_package_present", "package": "vendor/missing"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "is missing" in msg

    def test_npm_success_message_says_is_present(self, tmp_path: Path) -> None:
        """On success, npm message must say 'is present'."""
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({"dependencies": {"express": "^4.0"}}), encoding="utf-8")
        assertion = {"type": "npm_package_present", "package": "express"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "is present" in msg

    def test_npm_failure_message_says_is_missing(self, tmp_path: Path) -> None:
        """On failure, npm message must say 'is missing'."""
        pkg = tmp_path / "package.json"
        pkg.write_text(json.dumps({"dependencies": {}}), encoding="utf-8")
        assertion = {"type": "npm_package_present", "package": "nothere"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "is missing" in msg


class TestEvaluateGlobMessages:
    def test_no_match_message_contains_matched_no_paths(self, tmp_path: Path) -> None:
        """The failure message from glob_exists should say 'matched no paths'."""
        assertion = {"type": "glob_exists", "glob": "**/*.rs"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "matched no paths" in msg

    def test_success_message_contains_at_least_one_path(self, tmp_path: Path) -> None:
        """The success message from glob_exists should say 'matched at least one path'."""
        (tmp_path / "hello.py").write_text("x", encoding="utf-8")
        assertion = {"type": "glob_exists", "glob": "*.py"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "matched at least one path" in msg


class TestLookupJsonPathArrayOnNonList:
    def test_array_index_into_string_returns_false(self, tmp_path: Path) -> None:
        """If a path step tries to index into a value that is not a list,
        _lookup_json_path returns None, making json_path_exists return False."""
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"name": "hello"}), encoding="utf-8")
        # "name" resolves to the string "hello"; then [0] tries to index a string
        assertion = {
            "type": "json_path_exists",
            "path": str(f),
            "json_path": "name[0]",
        }
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False

    def test_json_path_relative_path_resolved_from_base_dir(
        self, tmp_path: Path
    ) -> None:
        """json_path_exists resolves relative paths against base_dir."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"port": 8080}), encoding="utf-8")
        assertion = {
            "type": "json_path_exists",
            "path": "config.json",
            "json_path": "port",
        }
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — file_regex_absent with invalid regex
# ---------------------------------------------------------------------------


class TestFileRegexAbsentInvalidRegex:
    def test_invalid_regex_in_file_regex_absent_returns_false(self, tmp_path: Path) -> None:
        """An invalid regex pattern in file_regex_absent must return False and
        include an error hint in the message (same guard as file_regex)."""
        f = tmp_path / "app.py"
        f.write_text("some content\n", encoding="utf-8")
        assertion = {
            "type": "file_regex_absent",
            "path": str(f),
            "pattern": "[broken(",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "Invalid regex" in msg or "regex" in msg.lower()


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — json_path_exists message format
# ---------------------------------------------------------------------------


class TestJsonPathExistsMessageFormat:
    def test_success_message_says_exists(self, tmp_path: Path) -> None:
        """On success the json_path_exists message must include 'exists'."""
        f = tmp_path / "c.json"
        f.write_text('{"key": "val"}', encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "key"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "exists" in msg

    def test_failure_message_says_does_not_exist(self, tmp_path: Path) -> None:
        """On failure the json_path_exists message must include 'does not exist'."""
        f = tmp_path / "c.json"
        f.write_text('{"key": "val"}', encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "missing"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "does not exist" in msg


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — empty id alias raises ValueError
# ---------------------------------------------------------------------------


class TestNormalizeEmptyIdAlias:
    def test_empty_id_alias_raises_value_error(self) -> None:
        """The 'id' field is an alias for 'rule'. A blank id value must raise
        ValueError with a message about 'rule'."""
        a = {"type": "glob_exists", "glob": "*.py", "id": "   "}
        with pytest.raises(ValueError, match="rule"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# _lookup_json_path — nested object with embedded array index
# ---------------------------------------------------------------------------


class TestLookupJsonPathNestedArrayMixed:
    def test_object_then_array_index_then_key(self, tmp_path: Path) -> None:
        """json_path like 'data.items[0].name' traverses dict → list → dict."""
        f = tmp_path / "data.json"
        payload = {"data": {"items": [{"name": "first"}, {"name": "second"}]}}
        f.write_text(json.dumps(payload), encoding="utf-8")
        assertion = {
            "type": "json_path_exists",
            "path": str(f),
            "json_path": "data.items[0].name",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "exists" in msg

    def test_object_then_array_index_out_of_bounds(self, tmp_path: Path) -> None:
        """Traversal that fails at array index returns False."""
        f = tmp_path / "data.json"
        payload = {"data": {"items": ["only one"]}}
        f.write_text(json.dumps(payload), encoding="utf-8")
        assertion = {
            "type": "json_path_exists",
            "path": str(f),
            "json_path": "data.items[5]",
        }
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — non-dict input raises ValueError
# ---------------------------------------------------------------------------


class TestNonDictAssertionRaises:
    def test_string_input_raises_value_error(self) -> None:
        """Passing a bare string instead of a dict must raise ValueError with
        a message indicating the field must be an object."""
        with pytest.raises(ValueError, match="must be an object"):
            normalize_workspace_assertion("glob_exists", "assert[0]")

    def test_list_input_raises_value_error(self) -> None:
        """A list is not a valid assertion dict — must raise ValueError."""
        with pytest.raises(ValueError, match="must be an object"):
            normalize_workspace_assertion(["glob_exists"], "assert[0]")


# ---------------------------------------------------------------------------
# json_path_exists — root-level JSON array with array-index path
# ---------------------------------------------------------------------------


class TestJsonPathOnRootList:
    def test_root_list_array_index_found(self, tmp_path: Path) -> None:
        """When the JSON root is a list, a bare [N] path must resolve the
        element at that index and return True if it exists."""
        f = tmp_path / "list.json"
        f.write_text(json.dumps(["alpha", "beta", "gamma"]), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "[1]"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "exists" in msg

    def test_root_list_out_of_bounds_returns_false(self, tmp_path: Path) -> None:
        """Array index beyond the list length must return False."""
        f = tmp_path / "list.json"
        f.write_text(json.dumps(["only"]), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "[5]"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — invalid severity raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidSeverityRaises:
    def test_invalid_severity_raises_value_error(self) -> None:
        """A severity value not in {error, warning, info} must raise ValueError
        with a message mentioning 'severity'."""
        a = {
            "type": "glob_exists",
            "glob": "**/*.py",
            "severity": "critical",
        }
        with pytest.raises(ValueError, match="severity"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# glob_exists — absolute path that matches an existing file
# ---------------------------------------------------------------------------


class TestGlobExistsAbsolutePath:
    def test_absolute_glob_matches_existing_file(self, tmp_path: Path) -> None:
        """When the glob pattern is an absolute path pointing to an existing
        file, _eval_glob_exists must return True without prepending base_dir."""
        f = tmp_path / "target.txt"
        f.write_text("content", encoding="utf-8")
        assertion = {"type": "glob_exists", "glob": str(f)}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "matched at least one path" in msg


# ---------------------------------------------------------------------------
# file_not_contains with missing file
# ---------------------------------------------------------------------------


class TestFileNotContainsMissingFile:
    def test_file_not_contains_missing_file_returns_false(self, tmp_path: Path) -> None:
        """file_not_contains on a non-existent file must return False — the
        same path.exists() guard in _eval_file_content_assertion fires."""
        assertion = {
            "type": "file_not_contains",
            "path": str(tmp_path / "missing.py"),
            "pattern": "debug",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not found" in msg.lower()


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — whitespace-only type raises ValueError
# ---------------------------------------------------------------------------


class TestNormalizeWhitespaceOnlyType:
    def test_whitespace_only_type_raises_value_error(self) -> None:
        """A type field consisting only of whitespace strips to '' which is
        not in WORKSPACE_ASSERTION_TYPES → raises ValueError."""
        a = {"type": "   ", "glob": "*.py"}
        with pytest.raises(ValueError, match="not valid"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_missing_type_key_raises_value_error(self) -> None:
        """When the 'type' key is absent, get() returns '' which is not a
        valid assertion type → raises ValueError."""
        a = {"glob": "*.py"}
        with pytest.raises(ValueError, match="not valid"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# json_path_exists — path resolving to root object (dot-only path)
# ---------------------------------------------------------------------------


class TestJsonPathDotOnlyPath:
    def test_dot_only_path_resolves_to_root_and_returns_true(self, tmp_path: Path) -> None:
        """A json_path of '.' produces no tokens from the regex so
        _lookup_json_path returns the root object (a non-None dict)
        making json_path_exists pass."""
        f = tmp_path / "config.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "."}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# composer_package_present — require is a non-dict (list)
# ---------------------------------------------------------------------------


class TestComposerRequireNonDict:
    def test_require_is_list_not_dict_package_missing(self, tmp_path: Path) -> None:
        """When composer.json has 'require' set to a list (not a dict), the
        isinstance(require, dict) guard skips it and the package cannot be
        found, so the assertion returns False."""
        composer = tmp_path / "composer.json"
        composer.write_text(
            json.dumps({"require": ["vendor/pkg"]}), encoding="utf-8"
        )
        assertion = {"type": "composer_package_present", "package": "vendor/pkg"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "missing" in msg


# ---------------------------------------------------------------------------
# npm_package_present — one section is a non-dict value
# ---------------------------------------------------------------------------


class TestNpmSectionNonDict:
    def test_dependencies_is_list_section_skipped(self, tmp_path: Path) -> None:
        """When package.json has 'dependencies' as a list (not a dict), the
        isinstance(section, dict) guard skips it.  If no other sections have
        the package, the assertion returns False."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps({"dependencies": ["express"]}), encoding="utf-8"
        )
        assertion = {"type": "npm_package_present", "package": "express"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "missing" in msg

    def test_non_dict_section_with_dict_fallback_found(self, tmp_path: Path) -> None:
        """Even if 'dependencies' is a non-dict, 'devDependencies' being a
        proper dict still allows the package to be found there."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps({"dependencies": "not-a-dict", "devDependencies": {"jest": "^29"}}),
            encoding="utf-8",
        )
        assertion = {"type": "npm_package_present", "package": "jest"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# describe_workspace_assertion — rule is a non-string type
# ---------------------------------------------------------------------------


class TestDescribeNonStringRule:
    def test_numeric_rule_falls_through_to_type(self) -> None:
        """When 'rule' is a number (not a string), isinstance(rule, str) is
        False so describe falls through to return '{type}[{index}]'."""
        assertion = {"type": "glob_exists", "glob": "*.py", "rule": 42}
        desc = describe_workspace_assertion(assertion, 7)
        assert desc == "glob_exists[7]"


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — file_not_contains and file_regex
# path/pattern validation (mirrors file_contains / file_regex_absent gaps)
# ---------------------------------------------------------------------------


class TestNormalizeFileNotContainsRequirements:
    def test_file_not_contains_missing_path_raises(self) -> None:
        """file_not_contains is in _FILE_CONTENT_TYPES so missing 'path'
        must raise ValueError with 'path is required'."""
        a = {"type": "file_not_contains", "pattern": "debug"}
        with pytest.raises(ValueError, match="path is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_file_not_contains_missing_pattern_raises(self) -> None:
        """file_not_contains without 'pattern' must raise ValueError."""
        a = {"type": "file_not_contains", "path": "src/main.py"}
        with pytest.raises(ValueError, match="pattern is required"):
            normalize_workspace_assertion(a, "assert[0]")


class TestNormalizeFileRegexRequirements:
    def test_file_regex_missing_path_raises(self) -> None:
        """file_regex is in _FILE_CONTENT_TYPES so missing 'path'
        must raise ValueError with 'path is required'."""
        a = {"type": "file_regex", "pattern": r"def \w+"}
        with pytest.raises(ValueError, match="path is required"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_file_regex_missing_pattern_raises(self) -> None:
        """file_regex without 'pattern' must raise ValueError."""
        a = {"type": "file_regex", "path": "src/main.py"}
        with pytest.raises(ValueError, match="pattern is required"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — extra unknown keys are NOT preserved
# ---------------------------------------------------------------------------


class TestNormalizeUnknownFieldsRejected:
    """Unknown fields on workspace assertions must raise ValueError."""

    def test_single_unknown_field_raises(self) -> None:
        a = {
            "type": "glob_exists",
            "glob": "**/*.py",
            "negate": True,
        }
        with pytest.raises(ValueError, match="unknown field.*negate"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_multiple_unknown_fields_raises(self) -> None:
        a = {
            "type": "glob_exists",
            "glob": "**/*.py",
            "description": "extra",
            "enabled": True,
            "custom_field": "ignored",
        }
        with pytest.raises(ValueError, match="unknown field"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_unknown_field_on_file_contains(self) -> None:
        a = {
            "type": "file_contains",
            "path": "foo.py",
            "pattern": "hello",
            "negate": True,
        }
        with pytest.raises(ValueError, match="negate"):
            normalize_workspace_assertion(a, "assert[0]")

    def test_known_optional_fields_accepted(self) -> None:
        """message, severity, rule, task_id are all valid optional fields."""
        a = {
            "type": "file_contains",
            "path": "foo.py",
            "pattern": "hello",
            "message": "Check greeting",
            "severity": "warning",
            "rule": "GREET-001",
            "task_id": "impl",
        }
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["message"] == "Check greeting"
        assert result["severity"] == "warning"
        assert result["rule"] == "GREET-001"
        assert result["task_id"] == "impl"


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — file_regex_absent on a directory
# ---------------------------------------------------------------------------


class TestFileRegexAbsentDirectoryPath:
    def test_directory_path_fails_for_file_regex_absent(self, tmp_path: Path) -> None:
        """file_regex_absent on a directory path must return False with a
        'not a file' message — same path.is_file() guard as file_contains."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        assertion = {"type": "file_regex_absent", "path": str(sub), "pattern": r"eval\("}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not a file" in msg.lower()


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — npm_package_present with no recognized sections
# ---------------------------------------------------------------------------


class TestNpmPackageNoRecognizedSections:
    def test_no_recognized_sections_package_is_missing(self, tmp_path: Path) -> None:
        """When package.json has no keys that match any of the four recognized
        npm dependency sections (dependencies, devDependencies, peerDependencies,
        optionalDependencies), the packages set stays empty and the assertion
        returns False."""
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps({"name": "my-app", "version": "1.0.0", "scripts": {"test": "jest"}}),
            encoding="utf-8",
        )
        assertion = {"type": "npm_package_present", "package": "express"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "missing" in msg


# ---------------------------------------------------------------------------
# json_path_exists — falsy-but-not-None container values ([], {})
# ---------------------------------------------------------------------------


class TestJsonPathExistsContainerValues:
    def test_empty_list_at_path_returns_true(self, tmp_path: Path) -> None:
        """An empty list [] is not None, so json_path_exists must return True —
        _lookup_json_path returns [] which satisfies ``value is not None``."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"items": []}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "items"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "exists" in msg

    def test_empty_dict_at_path_returns_true(self, tmp_path: Path) -> None:
        """An empty dict {} is not None, so json_path_exists must return True."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"settings": {}}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "settings"}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — empty string for json_path field
# ---------------------------------------------------------------------------


class TestNormalizeEmptyStringJsonPath:
    def test_empty_json_path_string_raises_non_empty(self) -> None:
        """If json_path is present but set to an empty string, the non-empty
        guard in the key-copy loop fires before the 'json_path is required'
        check — raises ValueError mentioning 'non-empty'."""
        a = {"type": "json_path_exists", "path": "config.json", "json_path": ""}
        with pytest.raises(ValueError, match="non-empty"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# file_regex_absent — message format for pass and fail
# ---------------------------------------------------------------------------


class TestFileRegexAbsentMessageFormat:
    def test_success_message_contains_did_not_match(self, tmp_path: Path) -> None:
        """file_regex_absent pass (regex doesn't match) → message includes
        'did not match'."""
        f = tmp_path / "app.py"
        f.write_text("safe content\n", encoding="utf-8")
        assertion = {"type": "file_regex_absent", "path": str(f), "pattern": r"eval\("}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "did not match" in msg

    def test_failure_message_contains_matched(self, tmp_path: Path) -> None:
        """file_regex_absent fail (regex matches) → message includes 'matched'."""
        f = tmp_path / "app.py"
        f.write_text("eval(user_input)\n", encoding="utf-8")
        assertion = {"type": "file_regex_absent", "path": str(f), "pattern": r"eval\("}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "matched" in msg


# ---------------------------------------------------------------------------
# file_contains — multiline pattern matching
# ---------------------------------------------------------------------------


class TestFileContainsMultilinePattern:
    def test_multiline_substring_found(self, tmp_path: Path) -> None:
        """file_contains matches a multi-line substring spanning two lines."""
        f = tmp_path / "config.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        assertion = {"type": "file_contains", "path": str(f), "pattern": "line1\nline2"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_multiline_substring_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "config.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        assertion = {"type": "file_contains", "path": str(f), "pattern": "line1\nline4"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False


# ---------------------------------------------------------------------------
# json_path_exists — combined dot + array notation
# ---------------------------------------------------------------------------


class TestJsonPathCombinedNotation:
    def test_dot_then_array_then_dot(self, tmp_path: Path) -> None:
        """Complex path like 'data.items[0].name' traverses dict→list→dict."""
        f = tmp_path / "data.json"
        payload = {"data": {"items": [{"name": "first"}, {"name": "second"}]}}
        f.write_text(json.dumps(payload), encoding="utf-8")
        assertion = {
            "type": "json_path_exists",
            "path": str(f),
            "json_path": "data.items[0].name",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_array_in_middle_out_of_bounds(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        payload = {"data": {"items": [{"name": "only"}]}}
        f.write_text(json.dumps(payload), encoding="utf-8")
        assertion = {
            "type": "json_path_exists",
            "path": str(f),
            "json_path": "data.items[5].name",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False


# ---------------------------------------------------------------------------
# composer_package_present — package in both require and require-dev
# ---------------------------------------------------------------------------


class TestComposerPackageDuplicate:
    def test_package_in_both_require_and_require_dev(self, tmp_path: Path) -> None:
        """Package appears in both require and require-dev — still found."""
        f = tmp_path / "composer.json"
        payload = {
            "require": {"monolog/monolog": "^2.0"},
            "require-dev": {"monolog/monolog": "^2.0"},
        }
        f.write_text(json.dumps(payload), encoding="utf-8")
        assertion = {"type": "composer_package_present", "package": "monolog/monolog"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# glob_exists — empty directory with no matches
# ---------------------------------------------------------------------------


class TestGlobExistsEmptyDir:
    def test_glob_in_empty_dir_returns_false(self, tmp_path: Path) -> None:
        """A glob in a completely empty directory matches nothing."""
        assertion = {"type": "glob_exists", "glob": "*.py"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "matched no paths" in msg


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — file_regex and file_not_contains directory paths
# ---------------------------------------------------------------------------


class TestEvaluateFileRegexDirectoryPath:
    def test_directory_path_fails_for_file_regex(self, tmp_path: Path) -> None:
        """file_regex on a directory path must return False with a 'not a file'
        message — the path.is_file() guard in _eval_file_content_assertion."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        assertion = {"type": "file_regex", "path": str(sub), "pattern": r"def \w+"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not a file" in msg.lower()

    def test_directory_path_fails_for_file_not_contains(self, tmp_path: Path) -> None:
        """file_not_contains on a directory path must also return False — same
        path.is_file() guard fires for all four _FILE_CONTENT_TYPES."""
        sub = tmp_path / "another_dir"
        sub.mkdir()
        assertion = {"type": "file_not_contains", "path": str(sub), "pattern": "debug"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not a file" in msg.lower()


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — json_path_exists with empty json_path
# (bypasses normalization to test _lookup_json_path edge case)
# ---------------------------------------------------------------------------


class TestEvaluateJsonPathEmptyPath:
    def test_empty_json_path_returns_root_object_as_truthy(self, tmp_path: Path) -> None:
        """When json_path is '' (empty string, constructed directly to bypass
        normalize_workspace_assertion), _lookup_json_path returns the root
        object.  A non-None root makes json_path_exists return True."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": ""}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_empty_json_path_on_null_root_returns_false(self, tmp_path: Path) -> None:
        """A root JSON value of null deserializes to None.  _lookup_json_path
        with an empty path returns None, making json_path_exists return False."""
        f = tmp_path / "null.json"
        f.write_text("null", encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": ""}
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — valid severity passes through
# ---------------------------------------------------------------------------


class TestNormalizeValidSeverity:
    def test_valid_severity_preserved_in_output(self) -> None:
        """A severity value in {error, warning, info} must be preserved in the
        normalized output without raising."""
        a = {"type": "glob_exists", "glob": "**/*.py", "severity": "warning"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["severity"] == "warning"

    def test_all_valid_severities_accepted(self) -> None:
        """All three valid severity values must normalize without error."""
        for sev in ("error", "warning", "info"):
            a = {"type": "glob_exists", "glob": "*.py", "severity": sev}
            result = normalize_workspace_assertion(a, "assert[0]")
            assert result["severity"] == sev


# ---------------------------------------------------------------------------
# normalize_workspace_assertion — task_id field preserved
# ---------------------------------------------------------------------------


class TestNormalizeTaskIdField:
    def test_task_id_preserved_in_output(self) -> None:
        """The task_id field, when present and non-empty, must be included in
        the normalized output dict."""
        a = {"type": "glob_exists", "glob": "*.py", "task_id": "build-step"}
        result = normalize_workspace_assertion(a, "assert[0]")
        assert result["task_id"] == "build-step"

    def test_empty_task_id_raises(self) -> None:
        """An empty task_id value must raise ValueError."""
        a = {"type": "glob_exists", "glob": "*.py", "task_id": "  "}
        with pytest.raises(ValueError, match="task_id"):
            normalize_workspace_assertion(a, "assert[0]")


# ---------------------------------------------------------------------------
# _lookup_json_path — array-index token as first step into a dict root
# ---------------------------------------------------------------------------


class TestLookupJsonPathArrayIndexIntoDictRoot:
    def test_array_index_as_first_token_on_dict_root_returns_false(
        self, tmp_path: Path
    ) -> None:
        """A json_path starting with an array-index token (e.g. '[0]') applied
        to a dict root must return None from _lookup_json_path, making
        json_path_exists return False."""
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "[0]"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "does not exist" in msg


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — file_regex with relative path
# ---------------------------------------------------------------------------


class TestEvaluateFileRegexRelativePath:
    def test_file_regex_relative_path_resolved_from_base_dir(
        self, tmp_path: Path
    ) -> None:
        """A relative path in a file_regex assertion must be resolved against
        base_dir, not the cwd."""
        (tmp_path / "app.py").write_text("class MyApp:\n    pass\n", encoding="utf-8")
        assertion = {
            "type": "file_regex",
            "path": "app.py",
            "pattern": r"class\s+\w+:",
        }
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True

    def test_file_regex_absent_relative_path_resolved_from_base_dir(
        self, tmp_path: Path
    ) -> None:
        """A relative path in a file_regex_absent assertion must also be
        resolved against base_dir."""
        (tmp_path / "safe.py").write_text("result = 42\n", encoding="utf-8")
        assertion = {
            "type": "file_regex_absent",
            "path": "safe.py",
            "pattern": r"exec\(",
        }
        passed, _ = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — composer_package_present with no sections
# ---------------------------------------------------------------------------


class TestEvaluateComposerPackageNoSections:
    def test_composer_json_with_no_require_sections_package_is_missing(
        self, tmp_path: Path
    ) -> None:
        """A composer.json that has no 'require' or 'require-dev' keys means
        the packages set is empty and the assertion fails."""
        f = tmp_path / "composer.json"
        f.write_text(json.dumps({"name": "my/project", "version": "1.0.0"}), encoding="utf-8")
        assertion = {"type": "composer_package_present", "package": "vendor/lib"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "missing" in msg


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — npm_package_present with custom missing path
# ---------------------------------------------------------------------------


class TestEvaluateNpmCustomPathMissing:
    def test_custom_package_json_path_not_found_returns_false(
        self, tmp_path: Path
    ) -> None:
        """When a custom 'path' is given that doesn't exist, the assertion must
        fail with a message indicating the file was not found."""
        assertion = {
            "type": "npm_package_present",
            "path": "frontend/package.json",
            "package": "react",
        }
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "not found" in msg.lower() or "package.json" in msg.lower()


# ---------------------------------------------------------------------------
# evaluate_workspace_assertion — empty/unsupported type
# ---------------------------------------------------------------------------


class TestEvaluateUnsupportedType:
    def test_evaluate_empty_type_returns_unsupported(self, tmp_path: Path) -> None:
        """An assertion dict with an empty or unrecognized type string should
        return (False, 'Unsupported...') from evaluate_workspace_assertion."""
        assertion: dict[str, object] = {"type": ""}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "Unsupported" in msg or "unsupported" in msg.lower()

    def test_unknown_type_returns_false(self, tmp_path: Path) -> None:
        """A completely made-up assertion type should also return False."""
        assertion: dict[str, object] = {"type": "magic_unicorn_check"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "Unsupported" in msg


# ---------------------------------------------------------------------------
# _lookup_json_path via json_path_exists — consecutive array indices
# ---------------------------------------------------------------------------


class TestLookupJsonPathNestedArrays:
    def test_nested_array_consecutive_indices(self, tmp_path: Path) -> None:
        """json_path '[0][1]' on a nested array like [[10, 20], [30, 40]]
        should resolve to the element at outer[0][1] = 20."""
        f = tmp_path / "data.json"
        f.write_text(json.dumps([[10, 20], [30, 40]]), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "[0][1]"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is True
        assert "exists" in msg

    def test_nested_array_second_index_out_of_bounds(self, tmp_path: Path) -> None:
        """json_path '[0][5]' where inner array has only 2 elements should fail."""
        f = tmp_path / "data.json"
        f.write_text(json.dumps([[10, 20]]), encoding="utf-8")
        assertion = {"type": "json_path_exists", "path": str(f), "json_path": "[0][5]"}
        passed, msg = evaluate_workspace_assertion(assertion, tmp_path)
        assert passed is False
        assert "does not exist" in msg
