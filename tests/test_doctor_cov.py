from __future__ import annotations

import builtins
from typing import Any

import pytest

from maestro_cli.doctor import _engine_check_results
from maestro_cli.plugins import DoctorProbe, EnginePlugin


def _plugin(name: str) -> EnginePlugin:
    return EnginePlugin(
        name=name,
        build_command=lambda ctx: ([name], False),
        doctor_probe=DoctorProbe(executable=name),
    )


def _patch_plugin_layer(monkeypatch: Any, *, errors: dict[str, str]) -> None:
    """Stabilize the plugin-discovery boundary so optional-dep branches dominate."""
    monkeypatch.setattr("maestro_cli.doctor.discover_engine_plugins", lambda: {})
    monkeypatch.setattr("maestro_cli.doctor.plugin_discovery_errors", lambda: errors)
    monkeypatch.setattr("maestro_cli.doctor.supported_engine_names", lambda: [])
    monkeypatch.setattr("maestro_cli.doctor.get_engine_plugin", _plugin)
    monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)


def _make_failing_import(*, fail_names: set[str]) -> Any:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in fail_names:
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    return fake_import


# ===========================================================================
# `continue` when iterating discovery errors that include the
# reserved `__entry_points__` key. The key must be skipped, never surfaced
# as an `engine_*` check.
# ===========================================================================


class TestEntryPointsErrorSkipped:
    def test_entry_points_key_is_skipped_in_error_loop(self, monkeypatch: Any) -> None:
        # `_check_plugin_discovery` short-circuits to a single warn result when
        # `__entry_points__` is present, but `_engine_check_results` ALSO iterates
        # the full error dict afterwards. The reserved key must be `continue`d over
        # while a real plugin error key produces an engine_* warn.
        _patch_plugin_layer(
            monkeypatch,
            errors={"__entry_points__": "metadata unavailable", "acme": "bad plugin"},
        )

        results = _engine_check_results()
        names = [r[0] for r in results]

        # The reserved key never becomes an engine check.
        assert "engine___entry_points__" not in names
        # The real plugin error key does.
        assert "engine_acme" in names
        acme = next(r for r in results if r[0] == "engine_acme")
        assert acme[2] == "warn"
        assert "bad plugin" in acme[1]

    def test_only_entry_points_error_yields_no_engine_error_rows(
        self, monkeypatch: Any
    ) -> None:
        # With ONLY the reserved key present, the error loop must `continue`
        # on every iteration and never surface the reserved key as a check row.
        _patch_plugin_layer(
            monkeypatch,
            errors={"__entry_points__": "metadata unavailable"},
        )

        results = _engine_check_results()
        names = [r[0] for r in results]
        # The reserved key is skipped by the error loop: it never produces a
        # synthesized `engine_*` row. (Its detail still legitimately appears on
        # the dedicated `engine_plugins` discovery row, so we scope the check.)
        assert "engine___entry_points__" not in names
        error_loop_rows = [
            r for r in results if r[0] != "engine_plugins" and "metadata unavailable" in r[1]
        ]
        assert error_loop_rows == []


# ===========================================================================
#
# the `except ImportError` fallbacks for each optional dependency.
# Each is exercised by simulating that one import failing.
# ===========================================================================


class TestOptionalDependencyImportFailures:
    def _run_with_missing(self, monkeypatch: Any, pkg: str) -> list[Any]:
        _patch_plugin_layer(monkeypatch, errors={})
        monkeypatch.setattr(
            builtins, "__import__", _make_failing_import(fail_names={pkg})
        )
        return _engine_check_results()

    def test_textual_missing_reports_info(self, monkeypatch: Any) -> None:
        results = self._run_with_missing(monkeypatch, "textual")
        row = next(r for r in results if r[0] == "tui_dependency")
        assert row[2] == "info"
        assert "textual not installed" in row[1]

    def test_rich_missing_reports_info(self, monkeypatch: Any) -> None:
        results = self._run_with_missing(monkeypatch, "rich")
        row = next(r for r in results if r[0] == "live_dependency")
        assert row[2] == "info"
        assert "rich not installed" in row[1]

    def test_agui_missing_reports_info(self, monkeypatch: Any) -> None:
        results = self._run_with_missing(monkeypatch, "ag_ui")
        row = next(r for r in results if r[0] == "agui_protocol")
        assert row[2] == "info"
        assert "ag-ui-protocol not installed" in row[1]

    def test_mcp_missing_reports_info(self, monkeypatch: Any) -> None:
        results = self._run_with_missing(monkeypatch, "mcp")
        row = next(r for r in results if r[0] == "mcp_protocol")
        assert row[2] == "info"
        assert "mcp not installed" in row[1]

    def test_opentelemetry_missing_reports_info(self, monkeypatch: Any) -> None:
        results = self._run_with_missing(monkeypatch, "opentelemetry")
        row = next(r for r in results if r[0] == "otel_protocol")
        assert row[2] == "info"
        assert "opentelemetry not installed" in row[1]

    def test_all_optional_deps_missing_at_once(self, monkeypatch: Any) -> None:
        # Drive every except-branch in a single pass to prove they coexist.
        _patch_plugin_layer(monkeypatch, errors={})
        monkeypatch.setattr(
            builtins,
            "__import__",
            _make_failing_import(
                fail_names={"textual", "rich", "ag_ui", "mcp", "opentelemetry"}
            ),
        )

        results = _engine_check_results()
        by_name = {r[0]: r for r in results}

        for check in (
            "tui_dependency",
            "live_dependency",
            "agui_protocol",
            "mcp_protocol",
            "otel_protocol",
        ):
            assert by_name[check][2] == "info"
            assert "not installed" in by_name[check][1]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
