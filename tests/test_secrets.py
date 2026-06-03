from __future__ import annotations

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.runners import _build_secret_values, _mask_secrets


# ---------------------------------------------------------------------------
# _build_secret_values
# ---------------------------------------------------------------------------


class TestBuildSecretValues:
    def test_explicit_secrets_from_plan_env(self) -> None:
        plan_env = {"MY_API_KEY": "abc123secret"}
        values = _build_secret_values(["MY_API_KEY"], False, plan_env, {})
        assert "abc123secret" in values

    def test_explicit_secrets_from_system_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_SYSTEM_TOKEN", "sys-token-xyz")
        values = _build_secret_values(["MY_SYSTEM_TOKEN"], False, {}, {})
        assert "sys-token-xyz" in values

    def test_explicit_prefers_env_over_os(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHARED_KEY", "os-value")
        plan_env = {"SHARED_KEY": "plan-value"}
        values = _build_secret_values(["SHARED_KEY"], False, plan_env, {})
        assert "plan-value" in values
        assert "os-value" not in values

    def test_auto_detects_key_patterns(self) -> None:
        plan_env = {
            "OPENAI_API_KEY": "key-value-1",
            "DB_PASSWORD": "pass-value-2",
            "GITHUB_TOKEN": "token-value-3",
        }
        values = _build_secret_values([], True, plan_env, {})
        assert "key-value-1" in values
        assert "pass-value-2" in values
        assert "token-value-3" in values

    def test_auto_skips_short_values(self) -> None:
        plan_env = {"SOME_KEY": "ab", "ANOTHER_TOKEN": "x"}
        values = _build_secret_values([], True, plan_env, {})
        assert "ab" not in values
        assert "x" not in values

    def test_auto_checks_task_env(self) -> None:
        task_env = {"TASK_SECRET": "task-secret-value"}
        values = _build_secret_values([], True, {}, task_env)
        assert "task-secret-value" in values

    def test_auto_merges_plan_and_task_env(self) -> None:
        plan_env = {"PLAN_API_KEY": "plan-key-val"}
        task_env = {"TASK_TOKEN": "task-tok-val"}
        values = _build_secret_values([], True, plan_env, task_env)
        assert "plan-key-val" in values
        assert "task-tok-val" in values

    def test_empty_secrets_list(self) -> None:
        values = _build_secret_values([], False, {"PLAIN": "value"}, {})
        assert values == set()

    def test_explicit_missing_key_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        values = _build_secret_values(["NONEXISTENT_VAR"], False, {}, {})
        assert values == set()

    def test_auto_non_matching_env_skipped(self) -> None:
        plan_env = {"PLAIN_VAR": "plain-value", "ANOTHER_VAR": "another-value"}
        values = _build_secret_values([], True, plan_env, {})
        assert "plain-value" not in values
        assert "another-value" not in values


# ---------------------------------------------------------------------------
# _mask_secrets
# ---------------------------------------------------------------------------


class TestMaskSecrets:
    def test_masks_single_value(self) -> None:
        result = _mask_secrets("output contains mysecret123 here", {"mysecret123"})
        assert result == "output contains *** here"

    def test_masks_multiple_values(self) -> None:
        result = _mask_secrets("token=abc123 key=xyz789", {"abc123", "xyz789"})
        assert "abc123" not in result
        assert "xyz789" not in result
        assert "***" in result

    def test_masks_longest_first(self) -> None:
        # "secretlong" contains "secret" — masking longest first avoids partial replacement
        result = _mask_secrets("value=secretlong", {"secret", "secretlong"})
        assert result == "value=***"
        assert "secret" not in result

    def test_no_secrets_passthrough(self) -> None:
        original = "no secrets here"
        result = _mask_secrets(original, set())
        assert result == original

    def test_masks_in_multiline(self) -> None:
        text = "line1: token=abc\nline2: still abc\nline3: clean"
        result = _mask_secrets(text, {"abc"})
        assert "abc" not in result
        assert "line1: token=***" in result
        assert "line2: still ***" in result
        assert "line3: clean" in result

    def test_masks_all_occurrences(self) -> None:
        result = _mask_secrets("abc abc abc", {"abc"})
        assert result == "*** *** ***"


# ---------------------------------------------------------------------------
# Loader integration
# ---------------------------------------------------------------------------

_SECRETS_LIST_PLAN = """\
version: 1
name: test-secrets-list
secrets:
  - MY_API_KEY
  - DB_PASSWORD
tasks:
  - id: t1
    command: echo hello
"""

_SECRETS_AUTO_PLAN = """\
version: 1
name: test-secrets-auto
secrets: auto
tasks:
  - id: t1
    command: echo hello
"""

_SECRETS_INVALID_PLAN = """\
version: 1
name: test-secrets-invalid
secrets: 42
tasks:
  - id: t1
    command: echo hello
"""


class TestLoaderSecretsIntegration:
    def test_secrets_list_parsed(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_SECRETS_LIST_PLAN, encoding="utf-8")
        spec = load_plan(plan_file)
        assert "MY_API_KEY" in spec.secrets
        assert "DB_PASSWORD" in spec.secrets
        assert spec.secrets_auto is False

    def test_secrets_auto_parsed(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_SECRETS_AUTO_PLAN, encoding="utf-8")
        spec = load_plan(plan_file)
        assert spec.secrets_auto is True
        assert spec.secrets == []

    def test_secrets_invalid_type_raises_E024(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_SECRETS_INVALID_PLAN, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E024"):
            load_plan(plan_file)

    def test_secrets_defaults_merged_with_plan(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: test-merge
defaults:
  secrets:
    - DEFAULT_SECRET
secrets:
  - PLAN_SECRET
tasks:
  - id: t1
    command: echo hello
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(yaml, encoding="utf-8")
        spec = load_plan(plan_file)
        assert "DEFAULT_SECRET" in spec.secrets
        assert "PLAN_SECRET" in spec.secrets


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestPlanSpecToDict:
    def test_secrets_in_to_dict(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_SECRETS_LIST_PLAN, encoding="utf-8")
        spec = load_plan(plan_file)
        d = spec.to_dict()
        assert "secrets" in d
        assert "MY_API_KEY" in d["secrets"]
        assert "DB_PASSWORD" in d["secrets"]

    def test_secrets_auto_in_to_dict(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_SECRETS_AUTO_PLAN, encoding="utf-8")
        spec = load_plan(plan_file)
        d = spec.to_dict()
        assert d.get("secrets_auto") is True
