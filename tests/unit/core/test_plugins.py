from pathlib import Path
from typing import override

import pytest

from notarius_core.artifacts import (
    ArtifactTypeKey,
    ArtifactTypeSpec,
    InMemoryUnitOfWork,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import Node, NodeExecutionContext
from notarius_core.plugins import (
    Plugin,
    PluginRegistrationError,
    PluginRegistry,
    PluginRuntimeContext,
)
from notarius_storage import LocalFileObjectStore


class EmptyInput(NodeInput):
    pass


class EmptyOutput(NodeOutput):
    pass


class DefaultNode(Node[NoConfig, EmptyInput, EmptyOutput]):
    """Returns an empty output without runtime dependencies."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: EmptyInput,
        /,
    ) -> EmptyOutput:
        return EmptyOutput()


class ContextNode(Node[NoConfig, EmptyInput, EmptyOutput]):
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: EmptyInput,
        /,
    ) -> EmptyOutput:
        return EmptyOutput()


def runtime_context(tmp_path: Path) -> PluginRuntimeContext:
    return PluginRuntimeContext(
        workspace=tmp_path,
        uploads_dir=tmp_path / "uploads",
        storage=LocalFileObjectStore(tmp_path / "objects"),
        uow=InMemoryUnitOfWork(),
        bucket="test-artifacts",
    )


def test_node_decorator_records_plugin_metadata_and_docstring() -> None:
    plugin = Plugin(slug="example.tools", title="Example tools")
    decorated = plugin.node(
        operator_id="example.default",
        version=2,
        title="Default example",
    )(DefaultNode)

    registration = plugin.nodes[0]

    assert decorated is DefaultNode
    assert decorated.plugin_slug == "example.tools"
    assert decorated.title == "Default example"
    assert decorated.description == (
        "Returns an empty output without runtime dependencies."
    )
    assert registration.plugin_slug == "example.tools"
    assert registration.title == "Default example"
    assert registration.description == (
        "Returns an empty output without runtime dependencies."
    )
    assert registration.key == ("example.default", 2)


def test_registry_reports_operator_and_artifact_collisions() -> None:
    artifact = ArtifactTypeSpec(
        key=ArtifactTypeKey("example.value", 1),
        title="Example value",
    )
    first = Plugin(slug="example.first", title="First")
    first.node(operator_id="example.node", version=1, title="First node")(DefaultNode)
    first.register_artifact_type(artifact)

    conflicting_node = Plugin(slug="example.second", title="Second")
    conflicting_node.node(
        operator_id="example.node",
        version=1,
        title="Second node",
    )(ContextNode)

    registry = PluginRegistry()
    registry.install(first)

    with pytest.raises(
        PluginRegistrationError,
        match=(
            "Plugin 'example.second' operator example.node@1 conflicts with "
            "plugin 'example.first'"
        ),
    ):
        registry.install(conflicting_node)

    conflicting_artifact = Plugin(slug="example.third", title="Third")
    conflicting_artifact.register_artifact_type(artifact)
    with pytest.raises(
        PluginRegistrationError,
        match=(
            "Plugin 'example.third' artifact type example.value@1 is already installed"
        ),
    ):
        registry.install(conflicting_artifact)


def test_frozen_registry_rejects_late_installation() -> None:
    registry = PluginRegistry()
    registry.freeze()

    with pytest.raises(PluginRegistrationError, match="Plugin registry is frozen"):
        registry.install(Plugin(slug="example.late", title="Late"))


def test_registry_builds_default_and_context_factory_nodes(tmp_path: Path) -> None:
    plugin = Plugin(slug="example.builders", title="Builders")
    plugin.node(operator_id="example.default", version=1, title="Default")(DefaultNode)
    plugin.node(
        operator_id="example.context",
        version=1,
        title="Context",
        factory=lambda context: ContextNode(context.workspace),
    )(ContextNode)
    registry = PluginRegistry()
    registry.install(plugin)
    context = runtime_context(tmp_path)

    default_node = registry.build_node("example.default", 1, context)
    context_node = registry.build_node("example.context", 1, context)

    assert isinstance(default_node, DefaultNode)
    assert isinstance(context_node, ContextNode)
    assert context_node.workspace == tmp_path


def test_missing_factory_error_preserves_plugin_and_operator_context(
    tmp_path: Path,
) -> None:
    plugin = Plugin(slug="example.builders", title="Builders")
    plugin.node(operator_id="example.context", version=1, title="Context")(ContextNode)
    registry = PluginRegistry()
    registry.install(plugin)

    with pytest.raises(
        PluginRegistrationError,
        match=(
            "Plugin 'example.builders' operator example.context@1 requires an "
            "explicit node factory"
        ),
    ):
        registry.build_node("example.context", 1, runtime_context(tmp_path))
