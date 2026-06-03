from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Mapping, cast

if TYPE_CHECKING:
    from .models import ExecutionProfile, PlanSpec, TaskSpec

ENTRY_POINT_GROUP = "maestro_cli.engines"
BUILTIN_ENGINE_ORDER = (
    "codex",
    "claude",
    "gemini",
    "copilot",
    "qwen",
    "ollama",
    "llama",
)
BUILTIN_ENGINE_NAMES = frozenset(BUILTIN_ENGINE_ORDER)

__all__ = [
    "BUILTIN_ENGINE_NAMES",
    "CostExtraction",
    "DoctorProbe",
    "ENTRY_POINT_GROUP",
    "EngineCommandContext",
    "EnginePlugin",
    "PluginResolutionError",
    "clear_plugin_discovery_cache",
    "discover_engine_plugins",
    "get_engine_plugin",
    "has_engine",
    "plugin_discovery_errors",
    "register_builtin_engine",
    "supported_engine_names",
]

Command = str | list[str]
PricingTable = dict[str, tuple[float, float, float]]
EngineCommandBuilder = Callable[["EngineCommandContext"], tuple[Command, bool]]
PricingTableLoader = Callable[[], PricingTable]
PricingModelResolver = Callable[[str | None, list[str]], str | None]
DefaultModelResolver = Callable[["PlanSpec"], str | None]
CostExtractor = Callable[[Path, str | None], "CostExtraction"]
BuiltinRegistryLoader = Callable[[], None]


class PluginResolutionError(RuntimeError):
    """Raised when an engine plugin is missing or cannot be loaded safely."""


@dataclass(frozen=True)
class EngineCommandContext:
    plan: PlanSpec
    task: TaskSpec
    workdir: Path
    prompt_text: str
    execution_profile: ExecutionProfile = "plan"
    retry_feedback: str | None = None


@dataclass(frozen=True)
class DoctorProbe:
    executable: str
    check_name: str | None = None
    install_hint: str | None = None

    def resolved_check_name(self, engine_name: str) -> str:
        return self.check_name or f"engine_{engine_name}"


@dataclass(frozen=True)
class CostExtraction:
    cost_usd: float | None = None
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_tokens: int = 0


@dataclass(frozen=True)
class EnginePlugin:
    name: str
    build_command: EngineCommandBuilder
    model_aliases: Mapping[str, str] = field(default_factory=dict)
    doctor_probe: DoctorProbe | None = None
    load_pricing_table: PricingTableLoader | None = None
    resolve_pricing_model: PricingModelResolver | None = None
    get_default_model: DefaultModelResolver | None = None
    extract_cost: CostExtractor | None = None

    def resolve_model(self, model: str | None) -> str | None:
        if model is None:
            return None
        return self.model_aliases.get(model, model)


_builtin_plugins: dict[str, EnginePlugin] = {}
_discovered_plugins: dict[str, EnginePlugin] | None = None
_discovery_errors: dict[str, str] | None = None
_entry_points_error: str | None = None
_builtin_registry_loader: BuiltinRegistryLoader | None = None


def register_builtin_engine(plugin: EnginePlugin) -> None:
    _validate_plugin(plugin, source=f"builtin engine '{plugin.name}'")
    _builtin_plugins[plugin.name] = plugin


def _set_builtin_engine_loader(loader: BuiltinRegistryLoader) -> None:
    global _builtin_registry_loader
    _builtin_registry_loader = loader


def clear_plugin_discovery_cache() -> None:
    global _discovered_plugins, _discovery_errors, _entry_points_error
    _discovered_plugins = None
    _discovery_errors = None
    _entry_points_error = None


def _ensure_builtin_plugins_registered() -> None:
    if _builtin_registry_loader is None and not _builtin_plugins:
        try:
            from . import runners as _runners  # noqa: F401
        except Exception:
            return
    if _builtin_registry_loader is not None:
        _builtin_registry_loader()


def supported_engine_names(*, refresh: bool = False) -> list[str]:
    _ensure_builtin_plugins_registered()
    discovered, _ = _discover_plugins(refresh=refresh)
    builtin_names = list(BUILTIN_ENGINE_ORDER)
    extra_builtins = sorted(
        name for name in _builtin_plugins
        if name not in BUILTIN_ENGINE_NAMES
    )
    custom_names = sorted(
        name for name in discovered
        if name not in BUILTIN_ENGINE_NAMES and name not in _builtin_plugins
    )
    return builtin_names + extra_builtins + custom_names


def discover_engine_plugins(
    *,
    include_builtins: bool = False,
    refresh: bool = False,
) -> dict[str, EnginePlugin]:
    """Return the current engine registry.

    By default this returns only successfully loaded custom entry-point plugins.
    Set ``include_builtins=True`` to include registered built-in engines in the
    returned mapping as well.
    """
    if include_builtins:
        _ensure_builtin_plugins_registered()
    discovered, _ = _discover_plugins(refresh=refresh)
    if not include_builtins:
        return dict(discovered)

    plugins = dict(_builtin_plugins)
    plugins.update(discovered)
    return plugins


def plugin_discovery_errors(*, refresh: bool = False) -> dict[str, str]:
    """Return actionable discovery/load failures keyed by engine name."""
    _, errors = _discover_plugins(refresh=refresh)
    out = dict(errors)
    if _entry_points_error:
        out["__entry_points__"] = _entry_points_error
    return out


def has_engine(name: str, *, refresh: bool = False) -> bool:
    _ensure_builtin_plugins_registered()
    if name in BUILTIN_ENGINE_NAMES or name in _builtin_plugins:
        return True
    discovered, _ = _discover_plugins(refresh=refresh)
    return name in discovered


def get_engine_plugin(name: str, *, refresh: bool = False) -> EnginePlugin:
    _ensure_builtin_plugins_registered()
    if name in _builtin_plugins:
        return _builtin_plugins[name]

    discovered, errors = _discover_plugins(refresh=refresh)
    plugin = discovered.get(name)
    if plugin is not None:
        return plugin

    if name in BUILTIN_ENGINE_NAMES:
        raise PluginResolutionError(
            f"Built-in engine '{name}' is not registered. This is an internal error."
        )

    if name in errors:
        raise PluginResolutionError(errors[name])

    if _entry_points_error:
        raise PluginResolutionError(_entry_points_error)

    supported = ", ".join(supported_engine_names(refresh=refresh))
    raise PluginResolutionError(
        f"Unsupported engine '{name}'. Supported engines: {supported}. "
        f"To add a custom engine, install a package exposing the "
        f"'{ENTRY_POINT_GROUP}' entry point named '{name}'."
    )


def _discover_plugins(
    *,
    refresh: bool = False,
) -> tuple[dict[str, EnginePlugin], dict[str, str]]:
    global _discovered_plugins, _discovery_errors, _entry_points_error

    if refresh:
        clear_plugin_discovery_cache()

    if _discovered_plugins is not None and _discovery_errors is not None:
        return _discovered_plugins, _discovery_errors

    discovered: dict[str, EnginePlugin] = {}
    errors: dict[str, str] = {}
    _entry_points_error = None

    try:
        entry_points = metadata.entry_points()
    except Exception as exc:
        _entry_points_error = (
            f"Could not inspect installed engine plugins in '{ENTRY_POINT_GROUP}': "
            f"{exc.__class__.__name__}: {exc}"
        )
        _discovered_plugins = discovered
        _discovery_errors = errors
        return discovered, errors

    candidates: Iterable[metadata.EntryPoint]
    if hasattr(entry_points, "select"):
        candidates = entry_points.select(group=ENTRY_POINT_GROUP)
    else:
        # Deprecated dict-like API (Python <3.12 compat) — cast for type safety
        raw_eps: Any = entry_points.get(ENTRY_POINT_GROUP, [])
        candidates = cast(list[metadata.EntryPoint], raw_eps)

    for entry_point in candidates:
        if entry_point.name in BUILTIN_ENGINE_NAMES or entry_point.name in _builtin_plugins:
            errors[entry_point.name] = (
                f"Engine plugin '{entry_point.name}' from '{ENTRY_POINT_GROUP}' conflicts "
                "with a built-in engine. Built-ins cannot be overridden via entry points; "
                "rename or remove the conflicting plugin package."
            )
            continue

        dist_name = getattr(getattr(entry_point, "dist", None), "name", None)
        source = (
            f"entry point '{entry_point.name}' from {dist_name}"
            if dist_name
            else f"entry point '{entry_point.name}'"
        )
        try:
            raw_plugin = _load_entry_point_plugin(entry_point)
            _validate_plugin(raw_plugin, source=source)
        except Exception as exc:
            errors[entry_point.name] = (
                f"Engine plugin '{entry_point.name}' could not be loaded from "
                f"'{ENTRY_POINT_GROUP}' ({source}): {exc.__class__.__name__}: {exc}. "
                "Reinstall or remove the plugin package, or fix the entry point target."
            )
            continue

        # _validate_plugin guarantees this is EnginePlugin
        if not isinstance(raw_plugin, EnginePlugin):
            continue
        plugin: EnginePlugin = raw_plugin

        if plugin.name != entry_point.name:
            errors[entry_point.name] = (
                f"Engine plugin '{entry_point.name}' from '{ENTRY_POINT_GROUP}' returned "
                f"name '{plugin.name}'. The entry point name and plugin name must match."
            )
            continue

        if plugin.name in discovered:
            errors[plugin.name] = (
                f"Multiple installed packages register engine '{plugin.name}' in "
                f"'{ENTRY_POINT_GROUP}'. Remove one of the conflicting plugins."
            )
            discovered.pop(plugin.name, None)
            continue

        discovered[plugin.name] = plugin

    _discovered_plugins = discovered
    _discovery_errors = errors
    return discovered, errors


def _load_entry_point_plugin(entry_point: metadata.EntryPoint) -> object:
    loaded = entry_point.load()
    if isinstance(loaded, EnginePlugin):
        return loaded
    if not callable(loaded):
        return loaded

    try:
        signature = inspect.signature(loaded)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "entry point target must be an EnginePlugin instance or a zero-argument factory"
        ) from exc
    required_params = [
        param for param in signature.parameters.values()
        if param.default is inspect.Signature.empty
        and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if required_params:
        raise TypeError(
            "entry point target must be an EnginePlugin instance or a zero-argument factory"
        )
    return loaded()


def _validate_plugin(plugin: object, *, source: str) -> None:
    if not isinstance(plugin, EnginePlugin):
        raise TypeError(
            f"{source} must resolve to EnginePlugin, got {type(plugin).__name__}"
        )
    if not plugin.name:
        raise ValueError(f"{source} must define a non-empty name")
    if not callable(plugin.build_command):
        raise TypeError(f"{source} must define a callable build_command")
    if not isinstance(plugin.model_aliases, Mapping):
        raise TypeError(f"{source} model_aliases must be a mapping")
    for alias, target in plugin.model_aliases.items():
        if not isinstance(alias, str) or not alias:
            raise TypeError(f"{source} model_aliases keys must be non-empty strings")
        if not isinstance(target, str) or not target:
            raise TypeError(f"{source} model_aliases values must be non-empty strings")
    if plugin.doctor_probe is not None and not isinstance(plugin.doctor_probe, DoctorProbe):
        raise TypeError(f"{source} doctor_probe must be a DoctorProbe")
    if plugin.doctor_probe is not None and not plugin.doctor_probe.executable:
        raise ValueError(f"{source} doctor_probe must define an executable")
    if plugin.load_pricing_table is not None and not callable(plugin.load_pricing_table):
        raise TypeError(f"{source} load_pricing_table must be callable")
    if plugin.resolve_pricing_model is not None and not callable(plugin.resolve_pricing_model):
        raise TypeError(f"{source} resolve_pricing_model must be callable")
    if plugin.get_default_model is not None and not callable(plugin.get_default_model):
        raise TypeError(f"{source} get_default_model must be callable")
    if plugin.extract_cost is not None and not callable(plugin.extract_cost):
        raise TypeError(f"{source} extract_cost must be callable")
