from collections.abc import Callable
from dataclasses import dataclass
from inspect import getdoc
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from notarius_core.artifacts import (
    ArtifactTypeSpec,
    UnitOfWorkPort,
)
from notarius_core.conversions import ArtifactConversion, ArtifactConversionKey
from notarius_core.nodes import Node
from notarius_core.ports.storage import FileStoragePort

if TYPE_CHECKING:
    from notarius_core.runtime.persistence import ArtifactOutputWriter
    from notarius_core.runtime.resolvers import Resolver


PLUGIN_ENTRY_POINT_GROUP = "notarius.plugins"


@dataclass(frozen=True, slots=True)
class PluginRuntimeContext:
    workspace: Path
    uploads_dir: Path
    storage: FileStoragePort
    uow: UnitOfWorkPort
    bucket: str
    storage_backend: str = "local"


NodeFactory: TypeAlias = Callable[
    [PluginRuntimeContext],
    Node[Any, Any, Any],
]
ResolverFactory: TypeAlias = Callable[
    [PluginRuntimeContext],
    "Resolver[object]",
]
WriterFactory: TypeAlias = Callable[
    [PluginRuntimeContext],
    "ArtifactOutputWriter",
]


@dataclass(frozen=True, slots=True)
class NodeRegistration:
    node_class: type[Node[Any, Any, Any]]
    factory: NodeFactory | None

    @property
    def plugin_slug(self) -> str:
        return self.node_class.plugin_slug

    @property
    def title(self) -> str:
        return self.node_class.title

    @property
    def description(self) -> str:
        return self.node_class.description

    @property
    def key(self) -> tuple[str, int]:
        return (
            self.node_class.operator_id,
            self.node_class.operator_version,
        )


class PluginRegistrationError(RuntimeError):
    pass


class UnknownOperatorError(LookupError):
    pass


class Plugin:
    def __init__(self, *, slug: str, title: str) -> None:
        if slug.strip() == "":
            raise PluginRegistrationError("Plugin slug must not be empty")
        if title.strip() == "":
            raise PluginRegistrationError(f"Plugin {slug!r} title must not be empty")
        self.slug = slug
        self.title = title
        self._nodes: dict[tuple[str, int], NodeRegistration] = {}
        self._artifact_types: dict[tuple[str, int], ArtifactTypeSpec] = {}
        self._artifact_conversions: dict[
            ArtifactConversionKey,
            ArtifactConversion[Any, Any],
        ] = {}
        self._resolver_factories: list[ResolverFactory] = []
        self._writer_factories: list[WriterFactory] = []

    def node[NodeT: Node[Any, Any, Any]](
        self,
        *,
        operator_id: str,
        version: int,
        title: str,
        factory: NodeFactory | None = None,
    ) -> Callable[[type[NodeT]], type[NodeT]]:
        if operator_id.strip() == "":
            raise PluginRegistrationError(
                f"Plugin {self.slug!r} node operator_id must not be empty"
            )
        if version < 1:
            raise PluginRegistrationError(
                f"Plugin {self.slug!r} node {operator_id!r} version must be positive"
            )
        if title.strip() == "":
            raise PluginRegistrationError(
                f"Plugin {self.slug!r} node {operator_id!r} title must not be empty"
            )

        def decorate(node_class: type[NodeT]) -> type[NodeT]:
            key = (operator_id, version)
            if key in self._nodes:
                raise PluginRegistrationError(
                    f"Plugin {self.slug!r} already declares operator "
                    f"{operator_id}@{version}"
                )
            node_class.operator_id = operator_id
            node_class.operator_version = version
            node_class.plugin_slug = self.slug
            node_class.title = title
            node_class.description = getdoc(node_class) or ""
            registered_class: type[Node[Any, Any, Any]] = node_class
            registration = NodeRegistration(
                node_class=registered_class,
                factory=factory,
            )
            self._nodes[key] = registration
            return node_class

        return decorate

    def register_artifact_type(self, artifact_type: ArtifactTypeSpec) -> None:
        key = (
            artifact_type.key.id,
            artifact_type.key.schema_version,
        )
        if key in self._artifact_types:
            raise PluginRegistrationError(
                f"Plugin {self.slug!r} already declares artifact type {key[0]}@{key[1]}"
            )
        self._artifact_types[key] = artifact_type

    def register_artifact_conversion[SourceT, TargetT](
        self,
        conversion: ArtifactConversion[SourceT, TargetT],
    ) -> None:
        if conversion.key in self._artifact_conversions:
            raise PluginRegistrationError(
                f"Plugin {self.slug!r} already declares artifact conversion "
                f"{conversion.key.id}@{conversion.key.version}"
            )
        self._artifact_conversions[conversion.key] = conversion

    def register_resolver(self, factory: ResolverFactory) -> None:
        self._resolver_factories.append(factory)

    def register_writer(self, factory: WriterFactory) -> None:
        self._writer_factories.append(factory)

    @property
    def nodes(self) -> tuple[NodeRegistration, ...]:
        return tuple(self._nodes.values())

    @property
    def artifact_types(self) -> tuple[ArtifactTypeSpec, ...]:
        return tuple(self._artifact_types.values())

    @property
    def artifact_conversions(self) -> tuple[ArtifactConversion[Any, Any], ...]:
        return tuple(self._artifact_conversions.values())

    @property
    def resolver_factories(self) -> tuple[ResolverFactory, ...]:
        return tuple(self._resolver_factories)

    @property
    def writer_factories(self) -> tuple[WriterFactory, ...]:
        return tuple(self._writer_factories)


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}
        self._nodes: dict[tuple[str, int], NodeRegistration] = {}
        self._artifact_types: dict[tuple[str, int], ArtifactTypeSpec] = {}
        self._artifact_conversions: dict[
            ArtifactConversionKey,
            ArtifactConversion[Any, Any],
        ] = {}
        self._resolver_factories: list[ResolverFactory] = []
        self._writer_factories: list[WriterFactory] = []
        self._frozen = False

    def install(self, plugin: Plugin) -> None:
        if self._frozen:
            raise PluginRegistrationError("Plugin registry is frozen")
        if plugin.slug in self._plugins:
            raise PluginRegistrationError(
                f"Plugin slug {plugin.slug!r} is already installed"
            )

        duplicate_nodes = [
            registration.key
            for registration in plugin.nodes
            if registration.key in self._nodes
        ]
        if duplicate_nodes:
            operator_id, version = duplicate_nodes[0]
            owner = self._nodes[(operator_id, version)].plugin_slug
            raise PluginRegistrationError(
                f"Plugin {plugin.slug!r} operator {operator_id}@{version} "
                f"conflicts with plugin {owner!r}"
            )

        duplicate_artifacts = [
            (artifact_type.key.id, artifact_type.key.schema_version)
            for artifact_type in plugin.artifact_types
            if (
                artifact_type.key.id,
                artifact_type.key.schema_version,
            )
            in self._artifact_types
        ]
        if duplicate_artifacts:
            artifact_id, schema_version = duplicate_artifacts[0]
            raise PluginRegistrationError(
                f"Plugin {plugin.slug!r} artifact type "
                f"{artifact_id}@{schema_version} is already installed"
            )

        duplicate_conversions = [
            conversion.key
            for conversion in plugin.artifact_conversions
            if conversion.key in self._artifact_conversions
        ]
        if duplicate_conversions:
            conversion_key = duplicate_conversions[0]
            raise PluginRegistrationError(
                f"Plugin {plugin.slug!r} artifact conversion "
                f"{conversion_key.id}@{conversion_key.version} is already installed"
            )

        self._plugins[plugin.slug] = plugin
        for registration in plugin.nodes:
            self._nodes[registration.key] = registration
        for artifact_type in plugin.artifact_types:
            key = (
                artifact_type.key.id,
                artifact_type.key.schema_version,
            )
            self._artifact_types[key] = artifact_type
        for conversion in plugin.artifact_conversions:
            self._artifact_conversions[conversion.key] = conversion
        self._resolver_factories.extend(plugin.resolver_factories)
        self._writer_factories.extend(plugin.writer_factories)

    def freeze(self) -> None:
        for conversion in self._artifact_conversions.values():
            endpoints = (
                ("source", conversion.source),
                ("target", conversion.target),
            )
            for endpoint_name, artifact_type in endpoints:
                key = (artifact_type.id, artifact_type.schema_version)
                if key in self._artifact_types:
                    continue
                raise PluginRegistrationError(
                    f"Artifact conversion {conversion.key.id}@"
                    f"{conversion.key.version} references {endpoint_name} artifact "
                    f"type {artifact_type.id}@{artifact_type.schema_version}, which "
                    "is not installed"
                )
        self._frozen = True

    @property
    def plugins(self) -> tuple[Plugin, ...]:
        return tuple(self._plugins.values())

    @property
    def nodes(self) -> tuple[NodeRegistration, ...]:
        return tuple(self._nodes.values())

    @property
    def artifact_types(self) -> tuple[ArtifactTypeSpec, ...]:
        return tuple(self._artifact_types.values())

    @property
    def artifact_conversions(self) -> tuple[ArtifactConversion[Any, Any], ...]:
        return tuple(self._artifact_conversions.values())

    def build_node(
        self,
        operator_id: str,
        operator_version: int,
        context: PluginRuntimeContext,
    ) -> Node[Any, Any, Any]:
        registration = self._nodes.get((operator_id, operator_version))
        if registration is None:
            raise UnknownOperatorError(
                f"Unknown operator {operator_id!r} at version {operator_version}"
            )
        if registration.factory is not None:
            return registration.factory(context)
        node_class = registration.node_class
        try:
            return node_class()
        except TypeError as exc:
            raise PluginRegistrationError(
                f"Plugin {registration.plugin_slug!r} operator "
                f"{operator_id}@{operator_version} requires an explicit node factory"
            ) from exc

    def build_resolvers(
        self,
        context: PluginRuntimeContext,
    ) -> tuple["Resolver[object]", ...]:
        return tuple(factory(context) for factory in self._resolver_factories)

    def build_writers(
        self,
        context: PluginRuntimeContext,
    ) -> tuple["ArtifactOutputWriter", ...]:
        return tuple(factory(context) for factory in self._writer_factories)
