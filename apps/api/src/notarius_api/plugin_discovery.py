from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points

from notarius_core.plugins import (
    PLUGIN_ENTRY_POINT_GROUP,
    Plugin,
    PluginOrigin,
    PluginRegistry,
)


class PluginDiscoveryError(RuntimeError):
    pass


def discover_plugins() -> tuple[Plugin, ...]:
    discovered: list[Plugin] = []
    candidates = sorted(
        entry_points(group=PLUGIN_ENTRY_POINT_GROUP),
        key=lambda entry_point: (
            entry_point.name,
            entry_point.value,
        ),
    )
    for entry_point in candidates:
        discovered.append(_load_plugin(entry_point))
    return tuple(discovered)


def build_plugin_registry(
    builtin_plugins: Iterable[Plugin],
    external_plugins: Iterable[Plugin] | None = None,
) -> PluginRegistry:
    registry = PluginRegistry()
    for plugin in builtin_plugins:
        registry.install(plugin, origin=PluginOrigin.BUILTIN)
    plugins = discover_plugins() if external_plugins is None else external_plugins
    for plugin in plugins:
        registry.install(plugin, origin=PluginOrigin.EXTERNAL)
    registry.freeze()
    return registry


def _load_plugin(entry_point: EntryPoint) -> Plugin:
    distribution = entry_point.dist.name if entry_point.dist is not None else "unknown"
    try:
        value = entry_point.load()
    except Exception as exc:
        raise PluginDiscoveryError(
            f"Failed to load plugin entry point {entry_point.name!r} from "
            f"distribution {distribution!r} ({entry_point.value})"
        ) from exc
    if not isinstance(value, Plugin):
        raise PluginDiscoveryError(
            f"Plugin entry point {entry_point.name!r} from distribution "
            f"{distribution!r} returned {type(value).__name__}, expected Plugin"
        )
    return value
