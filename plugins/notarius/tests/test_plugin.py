from grafy_core.plugins import Plugin, PluginRegistry
from grafy_plugin import PLUGIN


def test_notarius_plugin_contract_freezes() -> None:
    registry = PluginRegistry()
    registry.install(PLUGIN)
    registry.freeze()

    assert isinstance(PLUGIN, Plugin)
    assert PLUGIN.slug == "external.notarius"
    assert {registration.key for registration in PLUGIN.nodes} == {
        ("notarius.dataset.extract_structured", 1),
        ("notarius.dataset.to_table", 1),
    }
    assert {
        (artifact.key.id, artifact.key.schema_version)
        for artifact in PLUGIN.artifact_types
    } == {("notarius.extraction.dataset", 1)}
