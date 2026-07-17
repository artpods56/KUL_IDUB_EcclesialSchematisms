from importlib.metadata import EntryPoint
from unittest.mock import Mock

import pytest

from notarius_api import plugin_discovery
from notarius_api.plugin_discovery import PluginDiscoveryError, build_plugin_registry
from notarius_core.operators.arithmetic import INTEGER_VALUE
from notarius_core.operators.text import TEXT
from notarius_core.plugins import (
    PLUGIN_ENTRY_POINT_GROUP,
    Plugin,
    PluginOrigin,
    PluginRegistrationError,
)


def test_entry_point_loads_plugin_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = EntryPoint(
        name="text",
        value="notarius_core.operators.text:TEXT",
        group=PLUGIN_ENTRY_POINT_GROUP,
    )

    monkeypatch.setattr(
        plugin_discovery,
        "entry_points",
        Mock(return_value=(entry_point,)),
    )

    assert plugin_discovery.discover_plugins() == (TEXT,)


def test_entry_point_rejects_non_plugin_value_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = EntryPoint(
        name="integer-value",
        value="notarius_core.operators.arithmetic:INTEGER_VALUE",
        group=PLUGIN_ENTRY_POINT_GROUP,
    )

    monkeypatch.setattr(
        plugin_discovery,
        "entry_points",
        Mock(return_value=(entry_point,)),
    )

    with pytest.raises(
        PluginDiscoveryError,
        match=(
            "Plugin entry point 'integer-value' from distribution 'unknown' "
            "returned ArtifactTypeSpec, expected Plugin"
        ),
    ):
        plugin_discovery.discover_plugins()

    assert INTEGER_VALUE.key.id == "scalar.integer"


def test_entry_point_load_failure_preserves_entry_point_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_point = EntryPoint(
        name="missing",
        value="notarius_core.operators.text:DOES_NOT_EXIST",
        group=PLUGIN_ENTRY_POINT_GROUP,
    )

    monkeypatch.setattr(
        plugin_discovery,
        "entry_points",
        Mock(return_value=(entry_point,)),
    )

    with pytest.raises(
        PluginDiscoveryError,
        match=(
            "Failed to load plugin entry point 'missing' from distribution "
            r"'unknown' \(notarius_core.operators.text:DOES_NOT_EXIST\)"
        ),
    ) as raised:
        plugin_discovery.discover_plugins()

    assert isinstance(raised.value.__cause__, AttributeError)


def test_plugin_origins_follow_installation_path_and_registry_is_frozen() -> None:
    builtin = Plugin(slug="external.named-builtin", title="Builtin")
    external = Plugin(slug="builtin.named-external", title="External")

    registry = build_plugin_registry((builtin,), external_plugins=(external,))

    assert [(plugin.slug, plugin.origin) for plugin in registry.plugins] == [
        ("external.named-builtin", PluginOrigin.BUILTIN),
        ("builtin.named-external", PluginOrigin.EXTERNAL),
    ]
    with pytest.raises(PluginRegistrationError, match="Plugin registry is frozen"):
        registry.install(Plugin(slug="external.late", title="Late"))
