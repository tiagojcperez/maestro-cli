from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any

from .models import WORKSPACE_ASSERTION_TYPES

_FILE_CONTENT_TYPES = {
    "file_contains",
    "file_not_contains",
    "file_regex",
    "file_regex_absent",
}
_FILE_COUNT_TYPES = {"file_contains_count"}
_PATH_PATTERN_TYPES = _FILE_CONTENT_TYPES | _FILE_COUNT_TYPES
_PACKAGE_ASSERTION_TYPES = {"composer_package_present", "npm_package_present"}
_VALID_SEVERITIES = {"error", "warning", "info"}
_KNOWN_ASSERT_FIELDS: set[str] = {
    "type", "path", "pattern", "glob", "json_path", "package",
    "message", "severity", "rule", "id", "task_id",
    "count", "min_count",
}


def normalize_workspace_assertion(assertion: Any, field_name: str) -> dict[str, Any]:
    """Validate and normalize a workspace assertion spec.

    Raises ``ValueError`` on invalid input so callers can map it to either
    plan validation errors or audit findings.
    """
    if not isinstance(assertion, dict):
        raise ValueError(f"{field_name} must be an object")

    assertion_type = str(assertion.get("type", "")).strip()
    if assertion_type not in WORKSPACE_ASSERTION_TYPES:
        raise ValueError(
            f"{field_name}.type '{assertion_type}' is not valid. "
            f"Allowed: {sorted(WORKSPACE_ASSERTION_TYPES)}"
        )

    normalized: dict[str, Any] = {"type": assertion_type}
    for key in ("path", "pattern", "glob", "json_path", "package", "message", "severity"):
        value = assertion.get(key)
        if value is not None:
            text = str(value).strip()
            if not text:
                raise ValueError(f"{field_name}.{key} must be a non-empty string")
            normalized[key] = text
    for key in ("count", "min_count"):
        value = assertion.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}.{key} must be an integer >= 0") from exc
        if parsed < 0:
            raise ValueError(f"{field_name}.{key} must be an integer >= 0")
        normalized[key] = parsed

    rule_name = assertion.get("rule", assertion.get("id"))
    if rule_name is not None:
        text = str(rule_name).strip()
        if not text:
            raise ValueError(f"{field_name}.rule must be a non-empty string")
        normalized["rule"] = text

    task_id = assertion.get("task_id")
    if task_id is not None:
        text = str(task_id).strip()
        if not text:
            raise ValueError(f"{field_name}.task_id must be a non-empty string")
        normalized["task_id"] = text

    if "severity" in normalized and normalized["severity"] not in _VALID_SEVERITIES:
        raise ValueError(
            f"{field_name}.severity '{normalized['severity']}' is not valid. "
            f"Allowed: {sorted(_VALID_SEVERITIES)}"
        )

    if assertion_type in _PATH_PATTERN_TYPES and "path" not in normalized:
        raise ValueError(f"{field_name}.path is required for type '{assertion_type}'")
    if assertion_type in _PATH_PATTERN_TYPES and "pattern" not in normalized:
        raise ValueError(f"{field_name}.pattern is required for type '{assertion_type}'")
    if assertion_type == "file_contains_count":
        has_count = "count" in normalized
        has_min_count = "min_count" in normalized
        if has_count == has_min_count:
            raise ValueError(
                f"{field_name}: type 'file_contains_count' requires exactly one of "
                f"'count' or 'min_count'"
            )
    if assertion_type == "glob_exists" and "glob" not in normalized:
        raise ValueError(f"{field_name}.glob is required for type 'glob_exists'")
    if assertion_type == "json_path_exists":
        if "path" not in normalized:
            raise ValueError(f"{field_name}.path is required for type 'json_path_exists'")
        if "json_path" not in normalized:
            raise ValueError(
                f"{field_name}.json_path is required for type 'json_path_exists'"
            )
    if assertion_type in _PACKAGE_ASSERTION_TYPES and "package" not in normalized:
        raise ValueError(f"{field_name}.package is required for type '{assertion_type}'")

    unknown = set(assertion.keys()) - _KNOWN_ASSERT_FIELDS
    if unknown:
        raise ValueError(
            f"{field_name}: unknown field(s) {sorted(unknown)}. "
            f"Allowed: {sorted(_KNOWN_ASSERT_FIELDS)}"
        )

    return normalized


def evaluate_workspace_assertion(assertion: dict[str, Any], base_dir: Path) -> tuple[bool, str]:
    """Evaluate a workspace assertion against files rooted at ``base_dir``."""
    assertion_type = str(assertion.get("type", ""))

    if assertion_type == "glob_exists":
        return _eval_glob_exists(assertion, base_dir)
    if assertion_type == "json_path_exists":
        return _eval_json_path_exists(assertion, base_dir)
    if assertion_type == "composer_package_present":
        return _eval_composer_package_present(assertion, base_dir)
    if assertion_type == "npm_package_present":
        return _eval_npm_package_present(assertion, base_dir)
    if assertion_type in _PATH_PATTERN_TYPES:
        return _eval_file_content_assertion(assertion, base_dir)

    return False, f"Unsupported workspace assertion type: {assertion_type!r}"


def describe_workspace_assertion(assertion: dict[str, Any], index: int) -> str:
    rule = assertion.get("rule")
    if isinstance(rule, str) and rule.strip():
        return rule
    assertion_type = assertion.get("type")
    if isinstance(assertion_type, str) and assertion_type.strip():
        return f"{assertion_type}[{index}]"
    return f"assert[{index}]"


def _eval_file_content_assertion(assertion: dict[str, Any], base_dir: Path) -> tuple[bool, str]:
    path = _resolve_file_path(base_dir, assertion["path"])
    if not path.exists():
        return False, f"File not found: {path}"
    if not path.is_file():
        return False, f"Path is not a file: {path}"

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return False, f"Failed to read file {path}: {exc}"

    assertion_type = assertion["type"]
    pattern = assertion["pattern"]

    if assertion_type == "file_contains":
        passed = pattern in content
        return (
            passed,
            f"Substring {pattern!r} {'found' if passed else 'not found'} in {path.name}.",
        )

    if assertion_type == "file_contains_count":
        actual = content.count(pattern)
        if "count" in assertion:
            expected = assertion["count"]
            passed = actual == expected
            return (
                passed,
                f"Substring {pattern!r} occurred {actual} time(s) in {path.name}; "
                f"expected exactly {expected}.",
            )
        minimum = assertion["min_count"]
        passed = actual >= minimum
        return (
            passed,
            f"Substring {pattern!r} occurred {actual} time(s) in {path.name}; "
            f"expected at least {minimum}.",
        )

    if assertion_type == "file_not_contains":
        passed = pattern not in content
        return (
            passed,
            f"Substring {pattern!r} {'not found' if passed else 'found'} in {path.name}.",
        )

    try:
        matched = re.search(pattern, content, flags=re.MULTILINE) is not None
    except re.error as exc:
        return False, f"Invalid regex pattern {pattern!r}: {exc}"

    if assertion_type == "file_regex":
        return (
            matched,
            f"Regex {pattern!r} {'matched' if matched else 'did not match'} in {path.name}.",
        )

    return (
        not matched,
        f"Regex {pattern!r} {'did not match' if not matched else 'matched'} in {path.name}.",
    )


def _eval_glob_exists(assertion: dict[str, Any], base_dir: Path) -> tuple[bool, str]:
    pattern = assertion["glob"]
    search_pattern = pattern
    if not Path(pattern).is_absolute():
        search_pattern = str((base_dir / pattern).resolve())
    matches = glob.glob(search_pattern, recursive=True)
    passed = len(matches) > 0
    return (
        passed,
        f"Glob {pattern!r} {'matched at least one path' if passed else 'matched no paths'}.",
    )


def _eval_json_path_exists(assertion: dict[str, Any], base_dir: Path) -> tuple[bool, str]:
    path = _resolve_file_path(base_dir, assertion["path"])
    if not path.exists():
        return False, f"JSON file not found: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Failed to load JSON from {path}: {exc}"

    value = _lookup_json_path(payload, assertion["json_path"])
    passed = value is not None
    return (
        passed,
        f"JSON path {assertion['json_path']!r} "
        f"{'exists' if passed else 'does not exist'} in {path.name}.",
    )


def _eval_composer_package_present(assertion: dict[str, Any], base_dir: Path) -> tuple[bool, str]:
    path = _resolve_file_path(base_dir, assertion.get("path", "composer.json"))
    package = assertion["package"]
    if not path.exists():
        return False, f"composer.json not found: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Failed to load composer manifest {path}: {exc}"

    require = payload.get("require", {})
    require_dev = payload.get("require-dev", {})
    packages: set[str] = set()
    if isinstance(require, dict):
        packages.update(str(k) for k in require)
    if isinstance(require_dev, dict):
        packages.update(str(k) for k in require_dev)
    passed = package in packages
    return (
        passed,
        f"Composer package {package!r} "
        f"{'is present' if passed else 'is missing'} in {path.name}.",
    )


def _eval_npm_package_present(assertion: dict[str, Any], base_dir: Path) -> tuple[bool, str]:
    path = _resolve_file_path(base_dir, assertion.get("path", "package.json"))
    package = assertion["package"]
    if not path.exists():
        return False, f"package.json not found: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Failed to load npm manifest {path}: {exc}"

    packages: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        section = payload.get(key, {})
        if isinstance(section, dict):
            packages.update(str(k) for k in section)
    passed = package in packages
    return (
        passed,
        f"NPM package {package!r} "
        f"{'is present' if passed else 'is missing'} in {path.name}.",
    )


def _resolve_file_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _lookup_json_path(payload: object, path: str) -> object | None:
    current = payload
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    for token in tokens:
        if token.startswith("[") and token.endswith("]"):
            if not isinstance(current, list):
                return None
            index = int(token[1:-1])
            if index >= len(current):
                return None
            current = current[index]
            continue
        if not isinstance(current, dict):
            return None
        if token not in current:
            return None
        current = current[token]
    return current
