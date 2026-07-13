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


def test_explicit_external_plugins_are_installed_in_declared_order_and_frozen() -> None:
    builtin = Plugin(slug="builtin.example", title="Builtin")
    external = Plugin(slug="external.example", title="External")

    registry = build_plugin_registry((builtin,), external_plugins=(external,))

    assert [plugin.slug for plugin in registry.plugins] == [
        "builtin.example",
        "external.example",
    ]
    with pytest.raises(PluginRegistrationError, match="Plugin registry is frozen"):
        registry.install(Plugin(slug="external.late", title="Late"))
