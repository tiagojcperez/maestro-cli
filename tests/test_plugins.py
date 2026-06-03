from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from maestro_cli.plugins import (
    BUILTIN_ENGINE_ORDER,
    CostExtraction,
    DoctorProbe,
    EnginePlugin,
    PluginResolutionError,
    clear_plugin_discovery_cache,
    discover_engine_plugins,
    get_engine_plugin,
    has_engine,
    plugin_discovery_errors,
    register_builtin_engine,
    supported_engine_names,
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
        if group == "maestro_cli.engines":
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


class TestPluginDiscovery:
    def test_discover_engine_plugins_with_no_installed_plugins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )

        assert discover_engine_plugins() == {}
        assert plugin_discovery_errors() == {}
        assert supported_engine_names() == list(BUILTIN_ENGINE_ORDER)

        with pytest.raises(PluginResolutionError, match="Unsupported engine 'custom'"):
            get_engine_plugin("custom")

    def test_discover_engine_plugins_loads_entry_point_factories(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", lambda: _plugin("custom")),
            ]),
        )

        discovered = discover_engine_plugins()

        assert list(discovered) == ["custom"]
        assert get_engine_plugin("custom").name == "custom"
        assert "custom" in supported_engine_names()

    def test_discover_engine_plugins_loads_direct_entry_point_instances(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", _plugin("custom")),
            ]),
        )

        discovered = discover_engine_plugins()

        assert list(discovered) == ["custom"]
        assert discovered["custom"].name == "custom"

    def test_entry_point_name_must_match_plugin_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", lambda: _plugin("other-name")),
            ]),
        )

        errors = plugin_discovery_errors()

        assert "custom" in errors
        assert "must match" in errors["custom"]
        with pytest.raises(PluginResolutionError, match="must match"):
            get_engine_plugin("custom")

    def test_duplicate_entry_points_raise_actionable_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", lambda: _plugin("custom"), dist_name="a"),
                _FakeEntryPoint("custom", lambda: _plugin("custom"), dist_name="b"),
            ]),
        )

        assert discover_engine_plugins() == {}
        errors = plugin_discovery_errors()

        assert "custom" in errors
        assert "Multiple installed packages register engine 'custom'" in errors["custom"]
        with pytest.raises(PluginResolutionError, match="Multiple installed packages register engine 'custom'"):
            get_engine_plugin("custom")

    def test_broken_entry_point_reports_actionable_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("broken", lambda: 123),
            ]),
        )

        errors = plugin_discovery_errors()

        assert "broken" in errors
        assert "could not be loaded" in errors["broken"]
        with pytest.raises(PluginResolutionError, match="could not be loaded"):
            get_engine_plugin("broken")

    def test_invalid_plugin_metadata_reports_actionable_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint(
                    "custom",
                    lambda: EnginePlugin(
                        name="custom",
                        build_command=lambda ctx: (["custom"], False),
                        doctor_probe=object(),  # type: ignore[arg-type]
                    ),
                ),
            ]),
        )

        errors = plugin_discovery_errors()

        assert "custom" in errors
        assert "doctor_probe must be a DoctorProbe" in errors["custom"]
        with pytest.raises(PluginResolutionError, match="doctor_probe must be a DoctorProbe"):
            get_engine_plugin("custom")

    def test_builtin_name_conflict_reports_actionable_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("codex", lambda: _plugin("codex")),
            ]),
        )

        errors = plugin_discovery_errors()

        assert "codex" in errors
        assert "conflicts with a built-in engine" in errors["codex"]
        assert discover_engine_plugins() == {}
        assert "codex" in supported_engine_names()

    def test_discover_engine_plugins_include_builtins_merges_registry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", _plugin("custom")),
            ]),
        )

        result = discover_engine_plugins(include_builtins=True)

        # All six builtin names must appear
        for name in BUILTIN_ENGINE_ORDER:
            assert name in result
        # Custom plugin is also included
        assert "custom" in result

    def test_discover_engine_plugins_without_include_builtins_omits_builtins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", _plugin("custom")),
            ]),
        )

        result = discover_engine_plugins()

        assert "custom" in result
        for name in BUILTIN_ENGINE_ORDER:
            assert name not in result

    def test_refresh_clears_cache_and_rediscovers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call_count = 0

        def _entry_points() -> _FakeEntryPoints:
            nonlocal call_count
            call_count += 1
            return _FakeEntryPoints([])

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _entry_points)

        discover_engine_plugins()
        discover_engine_plugins()  # second call uses cache
        assert call_count == 1

        discover_engine_plugins(refresh=True)  # forces re-discovery
        assert call_count == 2

    def test_entry_points_api_failure_stored_as_discovery_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise() -> None:
            raise RuntimeError("importlib metadata unavailable")

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _raise)

        errors = plugin_discovery_errors()

        assert "__entry_points__" in errors
        assert "importlib metadata unavailable" in errors["__entry_points__"]

    def test_plugin_discovery_errors_refresh_clears_previous_entry_point_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call_count = 0

        def _flaky_entry_points() -> _FakeEntryPoints:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first call fails")
            return _FakeEntryPoints([])

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _flaky_entry_points)

        errors_first = plugin_discovery_errors()
        assert "__entry_points__" in errors_first

        errors_second = plugin_discovery_errors(refresh=True)
        assert "__entry_points__" not in errors_second

    def test_get_engine_plugin_surfaces_entry_points_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise() -> None:
            raise RuntimeError("totally broken")

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _raise)

        with pytest.raises(PluginResolutionError, match="totally broken"):
            get_engine_plugin("nonexistent")

    def test_factory_with_required_params_raises_actionable_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _factory_with_arg(required_arg: str) -> EnginePlugin:  # noqa: ARG001
            return _plugin("custom")

        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", _factory_with_arg),
            ]),
        )

        errors = plugin_discovery_errors()

        assert "custom" in errors
        assert "could not be loaded" in errors["custom"]

    def test_multiple_valid_plugins_all_discovered(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("alpha", _plugin("alpha"), dist_name="pkg-a"),
                _FakeEntryPoint("beta", _plugin("beta"), dist_name="pkg-b"),
            ]),
        )

        discovered = discover_engine_plugins()

        assert set(discovered) == {"alpha", "beta"}
        assert plugin_discovery_errors() == {}


class TestHasEngine:
    def test_builtin_engine_name_returns_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        for name in BUILTIN_ENGINE_ORDER:
            assert has_engine(name) is True

    def test_valid_custom_plugin_returns_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("acme", _plugin("acme")),
            ]),
        )
        assert has_engine("acme") is True

    def test_unknown_engine_name_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        assert has_engine("nonexistent") is False

    def test_broken_plugin_in_errors_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("broken", lambda: 42),
            ]),
        )
        assert has_engine("broken") is False

    def test_has_engine_refresh_forces_rediscovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call_count = 0

        def _entry_points() -> _FakeEntryPoints:
            nonlocal call_count
            call_count += 1
            return _FakeEntryPoints([])

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _entry_points)

        has_engine("nonexistent")
        has_engine("nonexistent")  # second call should use cache
        assert call_count == 1

        has_engine("nonexistent", refresh=True)  # forces re-discovery
        assert call_count == 2


class TestResolveModel:
    def test_known_alias_returns_mapped_value(self) -> None:
        plugin = EnginePlugin(
            name="myengine",
            build_command=lambda ctx: (["myengine"], False),
            model_aliases={"fast": "myengine-v1-fast", "slow": "myengine-v1-slow"},
        )
        assert plugin.resolve_model("fast") == "myengine-v1-fast"

    def test_unknown_model_returned_as_is(self) -> None:
        plugin = EnginePlugin(
            name="myengine",
            build_command=lambda ctx: (["myengine"], False),
            model_aliases={"fast": "myengine-v1-fast"},
        )
        assert plugin.resolve_model("custom-model-xyz") == "custom-model-xyz"

    def test_none_model_returns_none(self) -> None:
        plugin = _plugin("myengine")
        assert plugin.resolve_model(None) is None


class TestDoctorProbe:
    def test_resolved_check_name_with_explicit_check_name(self) -> None:
        probe = DoctorProbe(executable="myengine", check_name="myengine_available")
        assert probe.resolved_check_name("myengine") == "myengine_available"

    def test_resolved_check_name_without_explicit_check_name_uses_engine_prefix(self) -> None:
        probe = DoctorProbe(executable="myengine")
        assert probe.resolved_check_name("myengine") == "engine_myengine"

    def test_install_hint_is_optional_and_none_by_default(self) -> None:
        probe = DoctorProbe(executable="myengine")
        assert probe.install_hint is None

    def test_install_hint_set_to_string_is_returned(self) -> None:
        probe = DoctorProbe(executable="myengine", install_hint="pip install myengine")
        assert probe.install_hint == "pip install myengine"


class TestValidatePlugin:
    def test_not_engine_plugin_instance_raises_type_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", lambda: "not-a-plugin"),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "must resolve to EnginePlugin" in errors["custom"]

    def test_empty_name_raises_value_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", EnginePlugin(
                    name="",
                    build_command=lambda ctx: (["x"], False),
                )),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "non-empty name" in errors["custom"]

    def test_model_aliases_empty_key_raises_type_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", EnginePlugin(
                    name="custom",
                    build_command=lambda ctx: (["x"], False),
                    model_aliases={"": "something"},  # type: ignore[arg-type]
                )),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "model_aliases keys must be non-empty strings" in errors["custom"]

    def test_model_aliases_empty_value_raises_type_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", EnginePlugin(
                    name="custom",
                    build_command=lambda ctx: (["x"], False),
                    model_aliases={"fast": ""},
                )),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "model_aliases values must be non-empty strings" in errors["custom"]

    def test_doctor_probe_empty_executable_raises_value_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", EnginePlugin(
                    name="custom",
                    build_command=lambda ctx: (["x"], False),
                    doctor_probe=DoctorProbe(executable=""),
                )),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "doctor_probe must define an executable" in errors["custom"]

    @pytest.mark.parametrize("field_name,bad_value", [
        ("load_pricing_table", "not-callable"),
        ("resolve_pricing_model", 42),
        ("get_default_model", []),
        ("extract_cost", {}),
    ])
    def test_non_callable_optional_fields_raise_type_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        field_name: str,
        bad_value: Any,
    ) -> None:
        kwargs = {
            "name": "custom",
            "build_command": lambda ctx: (["x"], False),
            field_name: bad_value,
        }
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", EnginePlugin(**kwargs)),  # type: ignore[arg-type]
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "must be callable" in errors["custom"]


    def test_model_aliases_not_a_mapping_raises_type_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", EnginePlugin(
                    name="custom",
                    build_command=lambda ctx: (["x"], False),
                    model_aliases=["fast", "slow"],  # type: ignore[arg-type]
                )),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "model_aliases must be a mapping" in errors["custom"]

    def test_build_command_not_callable_reports_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", EnginePlugin(
                    name="custom",
                    build_command="not-a-callable",  # type: ignore[arg-type]
                )),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "must be callable" in errors["custom"] or "build_command" in errors["custom"]

    def test_supported_engine_names_custom_plugins_sorted_after_builtins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("zeta", _plugin("zeta"), dist_name="pkg-z"),
                _FakeEntryPoint("alpha", _plugin("alpha"), dist_name="pkg-a"),
            ]),
        )
        names = supported_engine_names()
        # All builtins appear first in canonical order
        assert names[: len(BUILTIN_ENGINE_ORDER)] == list(BUILTIN_ENGINE_ORDER)
        # Custom plugins appear at the end, sorted
        custom = names[len(BUILTIN_ENGINE_ORDER) :]
        assert custom == sorted(custom)
        assert "alpha" in custom
        assert "zeta" in custom

    def test_get_engine_plugin_builtin_name_not_registered_raises_internal_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Simulate a builtin engine name in BUILTIN_ENGINE_NAMES that is NOT in
        # _builtin_plugins (e.g., if the runners registration was skipped).
        monkeypatch.setattr("maestro_cli.plugins._builtin_plugins", {})
        monkeypatch.setattr(
            "maestro_cli.plugins._ensure_builtin_plugins_registered",
            lambda: None,
        )
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        with pytest.raises(PluginResolutionError, match="internal error"):
            get_engine_plugin("codex")


class TestRegisterBuiltinEngine:
    def test_valid_plugin_registered_and_accessible(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        plugin = _plugin("myengine")
        register_builtin_engine(plugin)

        assert has_engine("myengine")
        assert get_engine_plugin("myengine") is plugin

    def test_invalid_plugin_raises_on_register(self) -> None:
        # _validate_plugin checks isinstance first, but accesses plugin.name in the
        # source= argument before that — so a non-EnginePlugin raises AttributeError.
        with pytest.raises((TypeError, AttributeError)):
            register_builtin_engine("not-a-plugin")  # type: ignore[arg-type]

    def test_register_builtin_engine_with_empty_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty name"):
            register_builtin_engine(EnginePlugin(
                name="",
                build_command=lambda ctx: (["x"], False),
            ))

    def test_registering_same_name_twice_overwrites(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        first = _plugin("myengine")
        second = EnginePlugin(
            name="myengine",
            build_command=lambda ctx: (["myengine-v2"], False),
        )
        register_builtin_engine(first)
        register_builtin_engine(second)

        assert get_engine_plugin("myengine") is second


class TestSupportedEngineNames:
    def test_broken_plugins_excluded_from_supported_engine_names(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("broken", lambda: "bad"),
                _FakeEntryPoint("good", _plugin("good")),
            ]),
        )
        names = supported_engine_names()
        assert "good" in names
        assert "broken" not in names

    def test_extra_builtin_outside_canonical_order_sorted_after_standard_builtins(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        register_builtin_engine(_plugin("zzz_custom_builtin"))

        names = supported_engine_names()
        # Standard builtins appear first in canonical order
        assert names[: len(BUILTIN_ENGINE_ORDER)] == list(BUILTIN_ENGINE_ORDER)
        # Extra registered builtin appears after them, sorted
        assert "zzz_custom_builtin" in names
        assert names.index("zzz_custom_builtin") >= len(BUILTIN_ENGINE_ORDER)


class TestClearCache:
    def test_clear_plugin_discovery_cache_forces_rediscovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call_count = 0

        def _entry_points() -> _FakeEntryPoints:
            nonlocal call_count
            call_count += 1
            return _FakeEntryPoints([])

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _entry_points)

        discover_engine_plugins()
        assert call_count == 1

        clear_plugin_discovery_cache()
        discover_engine_plugins()
        assert call_count == 2

    def test_supported_engine_names_refresh_forces_rediscovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call_count = 0

        def _entry_points() -> _FakeEntryPoints:
            nonlocal call_count
            call_count += 1
            return _FakeEntryPoints([])

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _entry_points)

        supported_engine_names()
        supported_engine_names()  # cached
        assert call_count == 1

        supported_engine_names(refresh=True)
        assert call_count == 2


class TestBuiltinRegistryLoader:
    def test_builtin_registry_loader_is_called_when_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import maestro_cli.plugins as _mod

        called: list[int] = []

        def _loader() -> None:
            called.append(1)

        # Directly set the loader on the module and verify _ensure calls it.
        monkeypatch.setattr(_mod, "_builtin_registry_loader", _loader)
        _mod._ensure_builtin_plugins_registered()
        assert len(called) == 1

    def test_builtin_registry_loader_not_called_when_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import maestro_cli.plugins as _mod

        # With loader=None and _builtin_plugins already populated, no import occurs.
        monkeypatch.setattr(_mod, "_builtin_registry_loader", None)
        monkeypatch.setattr(_mod, "_builtin_plugins", {"codex": _plugin("codex")})
        # Should not raise and should not attempt any import.
        _mod._ensure_builtin_plugins_registered()  # no assertion needed — just must not raise


class TestEngineCommandContext:
    def test_default_execution_profile_is_plan(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock

        from maestro_cli.plugins import EngineCommandContext

        plan = MagicMock()
        task = MagicMock()
        ctx = EngineCommandContext(
            plan=plan,
            task=task,
            workdir=Path("/tmp"),
            prompt_text="hello",
        )
        assert ctx.execution_profile == "plan"
        assert ctx.retry_feedback is None
        assert ctx.prompt_text == "hello"

    def test_non_default_execution_profile_and_retry_feedback(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock

        from maestro_cli.plugins import EngineCommandContext

        ctx = EngineCommandContext(
            plan=MagicMock(),
            task=MagicMock(),
            workdir=Path("/tmp"),
            prompt_text="fix this bug",
            execution_profile="yolo",
            retry_feedback="previous attempt failed with exit 1",
        )
        assert ctx.execution_profile == "yolo"
        assert ctx.retry_feedback == "previous attempt failed with exit 1"

    def test_engine_command_context_is_frozen(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock

        from maestro_cli.plugins import EngineCommandContext

        ctx = EngineCommandContext(
            plan=MagicMock(),
            task=MagicMock(),
            workdir=Path("/tmp"),
            prompt_text="test",
        )
        with pytest.raises(Exception):
            ctx.prompt_text = "modified"  # type: ignore[misc]


class TestCostExtraction:
    def test_default_fields_are_none(self) -> None:
        ce = CostExtraction()
        assert ce.cost_usd is None
        assert ce.input_tokens is None
        assert ce.cached_tokens is None
        assert ce.output_tokens is None
        assert ce.cache_creation_tokens == 0

    def test_fields_set_correctly(self) -> None:
        ce = CostExtraction(
            cost_usd=0.005,
            input_tokens=1000,
            cached_tokens=200,
            output_tokens=500,
            cache_creation_tokens=100,
        )
        assert ce.cost_usd == 0.005
        assert ce.input_tokens == 1000
        assert ce.cached_tokens == 200
        assert ce.output_tokens == 500
        assert ce.cache_creation_tokens == 100

    def test_cost_extraction_is_frozen(self) -> None:
        ce = CostExtraction(cost_usd=1.0)
        with pytest.raises(Exception):
            ce.cost_usd = 2.0  # type: ignore[misc]


class TestEnginePluginFrozen:
    def test_engine_plugin_is_frozen(self) -> None:
        plugin = _plugin("myengine")
        with pytest.raises(Exception):
            plugin.name = "other"  # type: ignore[misc]


class TestLoadEntryPointPlugin:
    def test_factory_with_only_optional_params_is_called_successfully(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A callable where all params have defaults → no required_params →
        # _load_entry_point_plugin calls loaded() with no args and succeeds.
        def _factory_optional(extra: str = "default") -> EnginePlugin:  # noqa: ARG001
            return _plugin("custom")

        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", _factory_optional),
            ]),
        )
        discovered = discover_engine_plugins()
        assert "custom" in discovered
        assert discovered["custom"].name == "custom"

    def test_inspect_signature_error_reports_load_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Exercise the `except (TypeError, ValueError)` branch in
        # _load_entry_point_plugin when inspect.signature raises.
        import inspect as _inspect

        original_signature = _inspect.signature

        def _patched_signature(obj: object, **kwargs: object) -> object:
            if getattr(obj, "__name__", None) == "_uninspectable":
                raise TypeError("cannot inspect this callable")
            return original_signature(obj, **kwargs)  # type: ignore[arg-type]

        def _uninspectable() -> EnginePlugin:  # pragma: no cover
            return _plugin("custom")

        _uninspectable.__name__ = "_uninspectable"

        monkeypatch.setattr("maestro_cli.plugins.inspect.signature", _patched_signature)
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", _uninspectable),
            ]),
        )

        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "could not be loaded" in errors["custom"]

    def test_plugin_discovery_errors_refresh_clears_per_plugin_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # First call: broken plugin → per-plugin error stored.
        # Second call with refresh=True: plugin fixed → error gone.
        broken = [True]

        def _entry_points() -> _FakeEntryPoints:
            if broken[0]:
                return _FakeEntryPoints([_FakeEntryPoint("custom", lambda: "bad")])
            return _FakeEntryPoints([_FakeEntryPoint("custom", _plugin("custom"))])

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _entry_points)

        errors_first = plugin_discovery_errors()
        assert "custom" in errors_first

        broken[0] = False
        errors_second = plugin_discovery_errors(refresh=True)
        assert "custom" not in errors_second
        assert discover_engine_plugins()["custom"].name == "custom"

    def test_load_returning_non_callable_non_plugin_reports_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # load() returns a plain dict — not EnginePlugin, not callable
        # exercises the `if not callable(loaded): return loaded` branch
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", {"some": "dict"}),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custom" in errors
        assert "must resolve to EnginePlugin" in errors["custom"]

    def test_entry_point_without_dist_error_message_uses_name_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _NoDist:
            name = "nodist"

            def load(self) -> object:
                return "not-a-plugin"  # triggers TypeError in _validate_plugin

        class _FakeEPsNoDist:
            def select(self, *, group: str) -> list[_NoDist]:
                return [_NoDist()] if group == "maestro_cli.engines" else []

        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEPsNoDist(),
        )
        errors = plugin_discovery_errors()
        assert "nodist" in errors
        # Source string falls back to "entry point 'nodist'" (no "from dist-name")
        assert "entry point 'nodist'" in errors["nodist"]
        assert " from " not in errors["nodist"].split("could not be loaded")[0]

    def test_entry_point_conflicts_with_registered_builtin_reports_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # An entry point whose name matches a _builtin_plugins entry (not just BUILTIN_ENGINE_NAMES)
        # exercises the `entry_point.name in _builtin_plugins` branch in _discover_plugins
        register_builtin_engine(_plugin("custombuiltin"))
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custombuiltin", _plugin("custombuiltin")),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "custombuiltin" in errors
        assert "conflicts with a built-in engine" in errors["custombuiltin"]
        assert discover_engine_plugins() == {}

    def test_discover_engine_plugins_include_builtins_includes_registered_extra_builtin(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # discover_engine_plugins(include_builtins=True) merges _builtin_plugins,
        # so a registered extra builtin (not in BUILTIN_ENGINE_ORDER) must appear.
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        register_builtin_engine(_plugin("myextrabuiltin"))
        result = discover_engine_plugins(include_builtins=True)
        assert "myextrabuiltin" in result
        assert result["myextrabuiltin"].name == "myextrabuiltin"

    def test_dict_style_entry_points_api_without_select_method(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Exercise the `else: entry_points.get(group, [])` branch in _discover_plugins
        # for legacy importlib.metadata APIs that return a plain dict.
        ep = _FakeEntryPoint("custom", _plugin("custom"))

        class _DictStyleEntryPoints(dict):
            pass  # no .select — falls back to .get()

        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _DictStyleEntryPoints({"maestro_cli.engines": [ep]}),
        )
        discovered = discover_engine_plugins()
        assert "custom" in discovered
        assert discovered["custom"].name == "custom"

    def test_discover_engine_plugins_returns_independent_dict(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mutating the returned dict must not corrupt the internal cache.
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", _plugin("custom")),
            ]),
        )
        result1 = discover_engine_plugins()
        result1["injected"] = _plugin("injected")  # mutate the returned copy

        result2 = discover_engine_plugins()  # uses cache
        assert "injected" not in result2
        assert "custom" in result2

    def test_has_engine_returns_true_for_extra_registered_builtin(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A name registered via register_builtin_engine that is NOT in
        # BUILTIN_ENGINE_NAMES must still return True from has_engine().
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        register_builtin_engine(_plugin("myspecialengine"))
        assert has_engine("myspecialengine") is True

    def test_discover_engine_plugins_include_builtins_and_refresh_combined(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # include_builtins=True combined with refresh=True must clear the cache,
        # re-run discovery, and still include builtin engines in the result.
        call_count = 0

        def _entry_points() -> _FakeEntryPoints:
            nonlocal call_count
            call_count += 1
            return _FakeEntryPoints([_FakeEntryPoint("custom", _plugin("custom"))])

        monkeypatch.setattr("maestro_cli.plugins.metadata.entry_points", _entry_points)

        # Prime the cache with the first call (no builtins requested).
        discover_engine_plugins()
        assert call_count == 1

        # Now request with both flags — must re-discover AND include builtins.
        result = discover_engine_plugins(include_builtins=True, refresh=True)
        assert call_count == 2

        for name in BUILTIN_ENGINE_ORDER:
            assert name in result
        assert "custom" in result

    def test_get_engine_plugin_returns_same_object_on_repeated_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Two consecutive get_engine_plugin calls for the same plugin name must
        # return the identical EnginePlugin object (cache is not re-built).
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("custom", _plugin("custom")),
            ]),
        )
        first = get_engine_plugin("custom")
        second = get_engine_plugin("custom")
        assert first is second

    def test_plugin_discovery_errors_returns_independent_copy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mutating the returned dict must not corrupt the internal error cache.
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([
                _FakeEntryPoint("broken", lambda: "bad"),
            ]),
        )
        errors = plugin_discovery_errors()
        assert "broken" in errors
        errors.pop("broken")  # mutate the returned copy

        errors2 = plugin_discovery_errors()  # must still see the error
        assert "broken" in errors2

    def test_register_builtin_engine_with_all_optional_callables(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # register_builtin_engine must accept a fully-specified EnginePlugin that
        # provides all optional callable fields — validation must not raise.
        monkeypatch.setattr(
            "maestro_cli.plugins.metadata.entry_points",
            lambda: _FakeEntryPoints([]),
        )
        full_plugin = EnginePlugin(
            name="fullengine",
            build_command=lambda ctx: (["fullengine", ctx.prompt_text], False),
            model_aliases={"fast": "fullengine-fast"},
            doctor_probe=DoctorProbe(executable="fullengine", install_hint="pip install fullengine"),
            load_pricing_table=lambda: {"default": (0.1, 0.05, 0.2)},
            resolve_pricing_model=lambda model, aliases: model,
            get_default_model=lambda plan: "fast",
            extract_cost=lambda path, model: CostExtraction(cost_usd=0.001),
        )
        register_builtin_engine(full_plugin)

        assert has_engine("fullengine")
        retrieved = get_engine_plugin("fullengine")
        assert retrieved is full_plugin
        assert retrieved.resolve_model("fast") == "fullengine-fast"
        assert retrieved.resolve_model("unknown") == "unknown"
