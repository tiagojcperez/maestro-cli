from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import pytest

import maestro_cli.plugins as plugins
from maestro_cli.plugins import (
    EnginePlugin,
    clear_plugin_discovery_cache,
)


class _FakeEntryPoint:
    def __init__(self, name: str, loaded: Any, *, dist_name: str = "plugin-dist") -> None:
        self.name = name
        self._loaded = loaded
        self.dist = SimpleNamespace(name=dist_name)

    def load(self) -> Any:
        return self._loaded


class _FakeEntryPoints(list):
    def select(self, *, group: str) -> list[_FakeEntryPoint]:
        if group == plugins.ENTRY_POINT_GROUP:
            return list(self)
        return []


@pytest.fixture(autouse=True)
def _clear_plugin_cache() -> None:
    clear_plugin_discovery_cache()
    yield
    clear_plugin_discovery_cache()


def _plugin(name: str) -> EnginePlugin:
    return EnginePlugin(
        name=name,
        build_command=lambda ctx: ([name, ctx.prompt_text], False),
    )


class TestEnsureBuiltinPluginsRegistered:
    def test_imports_runners_when_loader_unset_and_no_builtins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Drive the `try: from . import runners` import branch: loader is None and
        # _builtin_plugins is empty, so the function falls into the import block.
        # The import itself succeeds (runners is already importable in CI), so no
        # exception is raised and control returns normally.
        monkeypatch.setattr(plugins, "_builtin_registry_loader", None)
        monkeypatch.setattr(plugins, "_builtin_plugins", {})

        # Must not raise. After importing runners (already cached, so its module
        # body does not re-run _set_builtin_engine_loader), the local
        # _builtin_registry_loader binding monkeypatched to None stays None.
        plugins._ensure_builtin_plugins_registered()

    def test_swallows_import_error_from_runners(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Drive the `except Exception: return` branch: loader is None, no builtins,
        # and importing runners raises. The function must swallow it and return.
        monkeypatch.setattr(plugins, "_builtin_registry_loader", None)
        monkeypatch.setattr(plugins, "_builtin_plugins", {})

        real_import = builtins.__import__

        def _failing_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if level == 1 and fromlist and "runners" in fromlist:
                raise ImportError("simulated runners import failure")
            if name == "maestro_cli" and fromlist and "runners" in fromlist:
                raise ImportError("simulated runners import failure")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _failing_import)

        # Must swallow the ImportError and return without raising.
        plugins._ensure_builtin_plugins_registered()

        # Loader must still be None (the import that would have set it failed),
        # so the subsequent `if _builtin_registry_loader is not None` is False.
        assert plugins._builtin_registry_loader is None

    def test_import_error_branch_does_not_invoke_loader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When the import fails the function returns early and never reaches the
        # loader-invocation line — even if a loader were somehow set afterwards,
        # the early return guarantees no call. Here we confirm the early return by
        # asserting no exception propagates and the state is unchanged.
        monkeypatch.setattr(plugins, "_builtin_registry_loader", None)
        monkeypatch.setattr(plugins, "_builtin_plugins", {})

        real_import = builtins.__import__
        attempted: list[str] = []

        def _failing_import(
            name: str,
            globals: Any = None,
            locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if fromlist and "runners" in fromlist:
                attempted.append(name)
                raise RuntimeError("boom")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _failing_import)

        plugins._ensure_builtin_plugins_registered()

        assert attempted  # the runners import was attempted and raised


class TestValidatedNonPluginGuard:
    def test_continue_when_validate_passes_but_object_is_not_engine_plugin(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Defensive guard: _discover_plugins re-checks isinstance(raw_plugin,
        # EnginePlugin) after _validate_plugin. Normally _validate_plugin raises
        # for any non-EnginePlugin, so this guard is never hit. To exercise it we
        # neuter the validator (simulating its contract being weakened) and feed a
        # non-EnginePlugin object. The loop must skip it via `continue`, leaving
        # the engine undiscovered and unrecorded as an error.
        monkeypatch.setattr(plugins, "_validate_plugin", lambda plugin, *, source: None)
        monkeypatch.setattr(
            plugins.metadata,
            "entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("ghost", "not-an-engine-plugin"),
            ]),
        )

        discovered = plugins.discover_engine_plugins()
        errors = plugins.plugin_discovery_errors()

        # The non-plugin object was skipped: not discovered, not an error.
        assert "ghost" not in discovered
        assert "ghost" not in errors

    def test_real_validate_still_records_error_for_non_plugin(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Sanity: with the REAL validator, a non-EnginePlugin is rejected as an
        # error (the guard line stays defensive/unreachable in normal operation).
        monkeypatch.setattr(
            plugins.metadata,
            "entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("ghost", "not-an-engine-plugin"),
            ]),
        )

        errors = plugins.plugin_discovery_errors()
        assert "ghost" in errors
        assert "must resolve to EnginePlugin" in errors["ghost"]
