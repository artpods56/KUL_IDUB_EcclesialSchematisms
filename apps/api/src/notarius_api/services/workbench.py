"""In-process execution backend for the workbench UI."""

import base64
import binascii
import json
import os
import re
from collections.abc import Mapping
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from PIL import Image as ImageModule
from PIL import ImageDraw
from pydantic import BaseModel, ConfigDict, TypeAdapter

from notarius_core.artifacts import (
    ArtifactFieldProjection,
    ArtifactObject,
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    ArtifactTypeSpec,
    InMemoryUnitOfWork,
)
from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.conversions import (
    MAX_ARTIFACT_CONVERSION_HOPS,
    ArtifactConversion,
    ArtifactConversionKey,
    conversion_runtime_types_are_compatible,
)
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.domain.modules import (
    MODULE_BOUNDARY_PORT,
    GraphModuleDefinition,
    GraphModuleDefinitionError,
    GraphModuleReference,
    GraphModuleReferenceError,
)
from notarius_core.domain.node_secrets import (
    InvalidNodeSecretDependenciesError,
    JsonValue,
    canonical_node_secret_dependencies,
)
from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphRevision
from notarius_core.nodes import (
    Node,
    NodeContractResolutionError,
    NodeExecutionContext,
    PortShape,
    ResolvedNodeContracts,
    resolve_node_contracts,
)
from notarius_core.operators.arithmetic import (
    IntegerValueOutputWriter,
    IntegerValueResolver,
)
from notarius_core.operators.modules import GraphModuleNode
from notarius_core.operators.text import TextValueOutputWriter, TextValueResolver
from notarius_core.plugins import (
    NodeCachePolicy,
    NodeRegistration,
    PluginRegistry,
    PluginRuntimeContext,
    UnknownOperatorError,
)
from notarius_core.ports.materialized_outputs import WorkbenchUnitOfWorkPort
from notarius_core.ports.modules import GraphModuleExecutionResult
from notarius_core.ports.node_secrets import (
    NodeSecretResolverPort,
    UnavailableNodeSecretResolver,
)
from notarius_core.ports.storage import FileStoragePort
from notarius_core.runtime.execution import NodeRuntime
from notarius_core.runtime.invocation import (
    InvocationError,
    InvocationMode,
    NodeInvocation,
    effective_input_shape,
    effective_output_shape,
    validate_invocation,
)
from notarius_core.runtime.materialization import (
    InputMaterializer,
    MaterializationProvenance,
)
from notarius_core.runtime.persistence import (
    ArtifactOutputWriter,
    ArtifactWriteContext,
    ArtifactWriterRegistry,
    OutputPersister,
    PersistedNodeOutput,
)
from notarius_core.runtime.resolvers import (
    Resolver,
    ResolverRegistry,
)
from notarius_storage import LocalFileObjectStore

from notarius_api.schemas.workbench import (
    ArtifactConversionRequest,
    ArtifactSummaryResponse,
    ArtifactTypeBindingModel,
    ArtifactTypeKeyResponse,
    FieldProjectionRequest,
    GraphMaterializationsResponse,
    PinnedOutputRequest,
    RunEdgeRequest,
    RunInputPlugRequest,
    RunNodeRequest,
    RunNodeResponse,
    RunPortOutputResponse,
    RunRequest,
    RunResponse,
    ImageUploadItemResponse,
)
from notarius_api.services.invocation_cache import PersistentInvocationCache

_WORKBENCH_BUCKET = "workbench-artifacts"
GRAPH_MODULE_PLUGIN_SLUG = "graph.module"
_SAMPLE_PAGE_TEXTS = (
    "PAGE {index}\nParochia Sancti Floriani\nAnno Domini 1846",
    "PAGE {index}\nBaptisatorum liber\nVilla Nova, folio {index}",
    "PAGE {index}\nIndex nominum\nSeries continua",
)


class WorkbenchGraphError(RuntimeError):
    pass


def _render_exception_chain(exception: BaseException) -> str:
    rendered: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen and len(rendered) < 12:
        seen.add(id(current))
        rendered.append(f"{type(current).__name__}: {current}")
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        current = None if current.__suppress_context__ else current.__context__
    return " <- caused by ".join(rendered)


def _default_workspace() -> Path:
    root = os.getenv(
        "NOTARIUS_WORKSPACE",
        ".notarius-artifacts/workbench",
    )
    return Path(root).resolve()


@dataclass(slots=True)
class _RunValue:
    """Output value of one node port during a workbench run."""

    value: ArtifactRef | ArtifactRefSequence


@dataclass(slots=True)
class _GraphExecution:
    response: RunResponse
    outputs: dict[str, dict[str, _RunValue]]


@dataclass(frozen=True, slots=True)
class GraphModuleCatalogEntry:
    definition: GraphModuleDefinition
    catalog_visible: bool


class WorkbenchService:
    def __init__(
        self,
        *,
        plugin_registry: PluginRegistry,
        workspace: Path | None = None,
        uow: WorkbenchUnitOfWorkPort | None = None,
        storage: FileStoragePort | None = None,
        storage_backend: str = "local",
        bucket: str = _WORKBENCH_BUCKET,
        saved_graphs: SavedGraphService | None = None,
        node_secrets: NodeSecretResolverPort | None = None,
    ) -> None:
        self._plugin_registry = plugin_registry
        self._workspace = (workspace or _default_workspace()).expanduser().resolve()
        self._uploads_dir = self._workspace / "uploads"
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._storage = storage or LocalFileObjectStore(self._workspace / "objects")
        self._storage_backend = storage_backend
        self._bucket = bucket
        self._uow = uow or InMemoryUnitOfWork()
        self._saved_graphs = saved_graphs
        self._node_secrets = node_secrets or UnavailableNodeSecretResolver()
        self._plugin_context = PluginRuntimeContext(
            workspace=self._workspace,
            uploads_dir=self._uploads_dir,
            storage=self._storage,
            uow=self._uow,
            bucket=self._bucket,
            storage_backend=self._storage_backend,
            node_secrets=self._node_secrets,
        )
        resolvers = [
            cast(Resolver[object], IntegerValueResolver(uow=self._uow)),
            cast(
                Resolver[object],
                TextValueResolver(uow=self._uow),
            ),
        ]
        resolvers.extend(plugin_registry.build_resolvers(self._plugin_context))
        self._resolvers = ResolverRegistry(resolvers)

        writers: list[ArtifactOutputWriter] = [
            IntegerValueOutputWriter(uow=self._uow),
            TextValueOutputWriter(uow=self._uow),
        ]
        writers.extend(plugin_registry.build_writers(self._plugin_context))
        self._writers = ArtifactWriterRegistry(writers)
        self._projectable_artifact_types = {
            artifact_type.key: artifact_type
            for artifact_type in plugin_registry.artifact_types
            if artifact_type.field_projections
        }
        self._artifact_types = {
            artifact_type.key for artifact_type in plugin_registry.artifact_types
        }
        self._artifact_conversions = {
            conversion.key: conversion
            for conversion in plugin_registry.artifact_conversions
        }
        self._runtime = NodeRuntime(
            materializer=InputMaterializer(self._resolvers),
            persister=OutputPersister(self._writers),
            invocation_cache=PersistentInvocationCache(
                unit_of_work=self._uow,
                storage=self._storage,
            ),
        )

    @property
    def plugin_registry(self) -> PluginRegistry:
        return self._plugin_registry

    async def list_graph_modules(self) -> list[GraphModuleCatalogEntry]:
        if self._saved_graphs is None:
            return []
        entries: list[GraphModuleCatalogEntry] = []
        for graph in await self._saved_graphs.list():
            for revision in await self._saved_graphs.list_revisions(graph.id):
                try:
                    definition = GraphModuleDefinition.from_saved_graph_revision(
                        revision
                    )
                except GraphModuleDefinitionError:
                    continue
                entries.append(
                    GraphModuleCatalogEntry(
                        definition=definition,
                        catalog_visible=revision.revision == graph.revision,
                    )
                )
        return entries

    async def _graph_module_definition(
        self,
        reference: GraphModuleReference,
    ) -> GraphModuleDefinition:
        if self._saved_graphs is None:
            raise WorkbenchGraphError(
                "Saved graph modules are not configured for this workbench"
            )
        revision = await self._saved_graphs.get_revision(
            reference.graph_id,
            reference.revision,
        )
        try:
            return GraphModuleDefinition.from_saved_graph_revision(revision)
        except GraphModuleDefinitionError as exc:
            raise WorkbenchGraphError(
                f"Saved graph {reference.graph_id} revision {reference.revision} "
                f"is not a valid module: {exc}"
            ) from exc

    async def _build_node(
        self,
        operator_id: str,
        operator_version: int,
    ) -> Node[Any, Any, Any]:
        try:
            module_reference = GraphModuleReference.try_from_operator_identity(
                operator_id,
                operator_version,
            )
        except GraphModuleReferenceError as exc:
            raise WorkbenchGraphError(str(exc)) from exc
        if module_reference is not None:
            definition = await self._graph_module_definition(module_reference)
            return GraphModuleNode(definition, self)
        try:
            return self._plugin_registry.build_node(
                operator_id,
                operator_version,
                self._plugin_context,
            )
        except UnknownOperatorError as exc:
            raise WorkbenchGraphError(str(exc)) from exc

    async def save_image_upload(
        self,
        filename: str,
        content_base64: str,
    ) -> ImageUploadItemResponse:
        try:
            content = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WorkbenchGraphError("Upload is not valid base64") from exc

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-") or "upload"
        path = self._uploads_dir / f"{uuid4().hex[:8]}-{safe_name}"
        path.write_bytes(content)
        return self._image_upload_item(path, filename=filename)

    async def create_sample_images(
        self,
        count: int,
    ) -> list[ImageUploadItemResponse]:
        items: list[ImageUploadItemResponse] = []
        for index in range(count):
            text = _SAMPLE_PAGE_TEXTS[index % len(_SAMPLE_PAGE_TEXTS)].format(
                index=index + 1
            )
            image = ImageModule.new("RGB", (420, 300), color="#f5f0e6")
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, 407, 287), outline="#b9ad98")
            draw.multiline_text((36, 48), text, fill="#463c2e", spacing=14)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            path = self._uploads_dir / f"{uuid4().hex[:8]}-sample-page.png"
            path.write_bytes(buffer.getvalue())
            items.append(
                self._image_upload_item(
                    path,
                    filename=f"sample-page-{index + 1}.png",
                )
            )
        return items

    def _image_upload_item(
        self,
        path: Path,
        filename: str,
    ) -> ImageUploadItemResponse:
        return ImageUploadItemResponse(
            upload_key=path.name,
            filename=filename,
            byte_size=path.stat().st_size,
        )

    async def run_graph(self, request: RunRequest) -> RunResponse:
        execution = await self._execute_graph(
            request,
            module_path=(),
            persist_materializations=True,
            validate_materialized_pins=True,
            raise_node_errors=False,
        )
        return execution.response

    async def execute_module(
        self,
        definition: GraphModuleDefinition,
        context: NodeExecutionContext,
        inputs: Mapping[str, ArtifactRef],
        /,
    ) -> GraphModuleExecutionResult:
        graph_id = definition.reference.graph_id
        graph_revision = definition.reference.revision
        graph_path_item = definition.reference.module_path_item
        if graph_path_item in context.module_path:
            rendered_path = " -> ".join((*context.module_path, graph_path_item))
            raise WorkbenchGraphError(
                f"Graph module cycle detected while entering {definition.name!r} "
                f"at revision {graph_revision}: {rendered_path}"
            )

        input_boundary_ids = {port.boundary_node_id for port in definition.input_ports}
        executed_nodes = [
            node
            for node in definition.document.nodes
            if node.id not in input_boundary_ids
        ]
        executed_node_ids = {node.id for node in executed_nodes}
        run_nodes = [
            RunNodeRequest(
                id=node.id,
                operator_id=node.operator_id,
                operator_version=node.operator_version,
                config=node.config_dict(),
                input_plugs=[
                    RunInputPlugRequest(id=plug.id, port=plug.port)
                    for plug in node.input_plugs
                ],
                artifact_type_bindings=[
                    ArtifactTypeBindingModel(
                        variable=binding.variable,
                        artifact_type=ArtifactTypeKeyResponse(
                            id=binding.artifact_type.id,
                            schema_version=binding.artifact_type.schema_version,
                        ),
                    )
                    for binding in node.artifact_type_bindings
                ],
            )
            for node in executed_nodes
        ]
        run_edges = [
            RunEdgeRequest(
                from_node=edge.from_node,
                from_port=edge.from_port,
                to_node=edge.to_node,
                to_port=edge.to_port,
                to_plug=edge.to_plug,
                projection=(
                    FieldProjectionRequest(path=list(edge.projection.path))
                    if edge.projection is not None
                    else None
                ),
                conversion_path=[
                    ArtifactConversionRequest(
                        id=conversion.id,
                        version=conversion.version,
                    )
                    for conversion in edge.conversion_path
                ],
                collection_mode=edge.collection_mode,
            )
            for edge in definition.document.edges
            if edge.to_node in executed_node_ids
        ]
        pinned_outputs = [
            PinnedOutputRequest(
                from_node=port.boundary_node_id,
                from_port=MODULE_BOUNDARY_PORT,
                value=inputs[port.name],
            )
            for port in definition.input_ports
        ]
        request = RunRequest(
            nodes=run_nodes,
            edges=run_edges,
            pinned_outputs=pinned_outputs,
            graph_id=graph_id,
            graph_revision=graph_revision,
            secret_graph_id=graph_id,
            secret_graph_revision=graph_revision,
        )
        execution = await self._execute_graph(
            request,
            module_path=(*context.module_path, graph_path_item),
            persist_materializations=False,
            validate_materialized_pins=False,
            raise_node_errors=True,
        )

        outputs: dict[str, ArtifactRef] = {}
        for port in definition.output_ports:
            boundary_outputs = execution.outputs.get(port.boundary_node_id)
            run_value = (
                boundary_outputs.get(MODULE_BOUNDARY_PORT)
                if boundary_outputs is not None
                else None
            )
            if run_value is None:
                raise WorkbenchGraphError(
                    f"Graph module {definition.name!r} revision {graph_revision} "
                    f"did not produce public output {port.name!r} at boundary "
                    f"node {port.boundary_node_id!r}"
                )
            if not isinstance(run_value.value, ArtifactRef):
                raise WorkbenchGraphError(
                    f"Graph module {definition.name!r} revision {graph_revision} "
                    f"public output {port.name!r} produced a sequence; module "
                    "boundary ports must be scalar"
                )
            outputs[port.name] = run_value.value
        return GraphModuleExecutionResult(outputs=outputs)

    async def _execute_graph(
        self,
        request: RunRequest,
        *,
        module_path: tuple[str, ...],
        persist_materializations: bool,
        validate_materialized_pins: bool,
        raise_node_errors: bool,
    ) -> _GraphExecution:
        submitted_secret_nodes = {
            node.id
            for node in request.nodes
            if any(
                registration.key == (node.operator_id, node.operator_version)
                and registration.secret_inputs
                for registration in self._plugin_registry.nodes
            )
        }
        if submitted_secret_nodes and request.secret_graph_id is None:
            rendered_node_ids = ", ".join(
                repr(node_id) for node_id in sorted(submitted_secret_nodes)
            )
            raise WorkbenchGraphError(
                "A saved secret graph context is required to run secret-bearing "
                f"nodes: {rendered_node_ids}"
            )
        materialization_graph: SavedGraph | SavedGraphRevision | None = None
        if request.graph_id is not None and request.graph_revision is not None:
            materialization_graph = await self._saved_graph_for_context(
                request.graph_id,
                request.graph_revision,
            )
            _validate_saved_graph_fragment(
                materialization_graph,
                request.nodes,
                request.edges,
            )
        secret_node_ids: set[str] = set()
        if (
            request.secret_graph_id is not None
            and request.secret_graph_revision is not None
        ):
            secret_graph = materialization_graph
            if secret_graph is None:
                secret_graph = await self._saved_graph_for_context(
                    request.secret_graph_id,
                    request.secret_graph_revision,
                )
            secret_node_ids = _validate_secret_graph_bindings(
                secret_graph,
                request.nodes,
                self._plugin_registry,
            )
        order = _topological_order(request.nodes, request.edges)
        pinned_outputs = _pinned_outputs_by_endpoint(
            request.nodes,
            request.edges,
            request.pinned_outputs,
        )
        if (
            validate_materialized_pins
            and request.graph_id is not None
            and request.graph_revision is not None
        ):
            await self._validate_materialized_pins(
                request.graph_id,
                request.graph_revision,
                pinned_outputs,
            )
        outputs = await self._resolve_pinned_outputs(pinned_outputs)
        nodes_by_id: dict[str, Node[Any, Any, Any]] = {}
        registrations_by_id: dict[str, NodeRegistration | None] = {}
        for node_request in order:
            nodes_by_id[node_request.id] = await self._build_node(
                node_request.operator_id,
                node_request.operator_version,
            )
            try:
                registrations_by_id[node_request.id] = (
                    self._plugin_registry.node_registration(
                        node_request.operator_id,
                        node_request.operator_version,
                    )
                )
            except UnknownOperatorError:
                registrations_by_id[node_request.id] = None
        artifact_type_bindings_by_node: dict[
            str,
            dict[str, ArtifactTypeKey],
        ] = {}
        resolved_contracts_by_node: dict[str, ResolvedNodeContracts] = {}
        for node_request in order:
            bindings = {
                binding.variable: binding.artifact_type.to_key()
                for binding in node_request.artifact_type_bindings
            }
            node = nodes_by_id[node_request.id]
            try:
                resolved_contracts = resolve_node_contracts(node, bindings)
            except NodeContractResolutionError as exc:
                raise WorkbenchGraphError(
                    f"Node {node_request.id!r} ({node.operator_id}@"
                    f"{node.operator_version}) has invalid artifact type "
                    f"bindings: {exc}"
                ) from exc
            for variable, artifact_type in bindings.items():
                if artifact_type in self._artifact_types:
                    continue
                raise WorkbenchGraphError(
                    f"Node {node_request.id!r} artifact type variable "
                    f"{variable!r} is bound to unavailable artifact type "
                    f"{artifact_type.id}@{artifact_type.schema_version}"
                )
            artifact_type_bindings_by_node[node_request.id] = bindings
            resolved_contracts_by_node[node_request.id] = resolved_contracts
        _validate_input_plugs(
            nodes_by_id,
            request.nodes,
            request.edges,
        )
        invocations_by_id = _derive_invocations(
            nodes_by_id,
            request.edges,
        )

        _validate_edges(
            nodes_by_id,
            resolved_contracts_by_node,
            invocations_by_id,
            request.edges,
            self._projectable_artifact_types,
            self._artifact_conversions,
            pinned_outputs,
        )
        failed: set[str] = set()
        node_runs: list[RunNodeResponse] = []
        run_id = uuid4()

        for node_request in order:
            upstream = {
                edge.from_node
                for edge in request.edges
                if edge.to_node == node_request.id
            }
            if upstream & failed:
                failed.add(node_request.id)
                node_runs.append(
                    RunNodeResponse(
                        node_id=node_request.id,
                        status="skipped",
                        error=None,
                        outputs=[],
                    )
                )
                continue

            try:
                node = nodes_by_id[node_request.id]
                inputs = await self._assemble_inputs(
                    node,
                    node_request,
                    request.edges,
                    outputs,
                    run_id,
                )
                node_context = NodeExecutionContext(
                    workflow_run_id=run_id,
                    node_run_id=uuid4(),
                    graph_id=request.graph_id,
                    graph_revision=request.graph_revision,
                    secret_graph_id=(
                        request.secret_graph_id
                        if node_request.id in secret_node_ids
                        else None
                    ),
                    secret_graph_revision=(
                        request.secret_graph_revision
                        if node_request.id in secret_node_ids
                        else None
                    ),
                    node_id=node_request.id,
                    module_path=module_path,
                )
                registration = registrations_by_id[node_request.id]
                cache_policy = (
                    registration.cache_policy
                    if registration is not None
                    else NodeCachePolicy.NEVER
                )
                opaque_secret_revisions: dict[str, str] = {}
                if (
                    cache_policy is NodeCachePolicy.EXACT
                    and registration is not None
                    and registration.secret_inputs
                ):
                    validated_config = (
                        registration.node_class.config_contract.model.model_validate(
                            node_request.config
                        ).model_dump(mode="json")
                    )
                    for secret_input in registration.secret_inputs:
                        secret_dependencies = {
                            dependency: cast(JsonValue, validated_config[dependency])
                            for dependency in secret_input.config_dependencies
                        }
                        opaque_secret_revisions[
                            secret_input.name
                        ] = await self._node_secrets.cache_revision(
                            graph_id=node_context.secret_graph_id,
                            graph_revision=node_context.secret_graph_revision,
                            node_id=node_context.node_id,
                            name=secret_input.name,
                            dependencies=secret_dependencies,
                        )
                result = await self._runtime.run_node(
                    node,
                    node_context,
                    inputs,
                    config=node_request.config,
                    invocation=invocations_by_id[node_request.id],
                    artifact_type_bindings=artifact_type_bindings_by_node[
                        node_request.id
                    ],
                    cache_policy=cache_policy,
                    opaque_secret_revisions=opaque_secret_revisions,
                )
            except Exception as exc:
                if raise_node_errors:
                    graph_context = "nested graph"
                    if request.graph_id is not None:
                        graph_context = (
                            f"graph {request.graph_id}@{request.graph_revision}"
                        )
                    raise WorkbenchGraphError(
                        f"{graph_context} node {node_request.id!r} "
                        f"({node_request.operator_id}@"
                        f"{node_request.operator_version}) failed"
                    ) from exc
                failed.add(node_request.id)
                node_runs.append(
                    RunNodeResponse(
                        node_id=node_request.id,
                        status="failed",
                        error=_render_exception_chain(exc),
                        outputs=[],
                    )
                )
                continue

            port_values = _port_values(node, result)
            outputs[node_request.id] = port_values
            node_run = RunNodeResponse(
                node_id=node_request.id,
                status="succeeded",
                error=None,
                outputs=[
                    await self._port_output_response(name, run_value)
                    for name, run_value in port_values.items()
                ],
            )
            if (
                persist_materializations
                and request.graph_id is not None
                and request.graph_revision is not None
            ):
                async with self._uow as uow:
                    await uow.materialized_outputs.upsert(
                        MaterializedNodeOutputs(
                            graph_id=request.graph_id,
                            graph_revision=request.graph_revision,
                            node_id=node_request.id,
                            workflow_run_id=run_id,
                            outputs={
                                name: run_value.value
                                for name, run_value in port_values.items()
                            },
                            materialized_at=datetime.now(UTC),
                        )
                    )
                    await uow.commit()
            node_runs.append(node_run)

        status: Literal["succeeded", "failed"] = "failed" if failed else "succeeded"
        return _GraphExecution(
            response=RunResponse(status=status, node_runs=node_runs),
            outputs=outputs,
        )

    async def get_graph_materializations(
        self,
        *,
        graph_id: UUID,
        graph_revision: int,
    ) -> GraphMaterializationsResponse:
        if graph_revision < 1:
            raise WorkbenchGraphError("Graph revision must be positive")
        await self._saved_graph_for_context(graph_id, graph_revision)
        async with self._uow as uow:
            materializations = await uow.materialized_outputs.list_for_graph(
                graph_id,
                graph_revision,
            )

        node_runs: list[RunNodeResponse] = []
        for materialization in materializations:
            accessible_outputs: list[RunPortOutputResponse] = []
            for port_name, value in materialization.outputs.items():
                if await self._run_value_is_accessible(value):
                    accessible_outputs.append(
                        await self._port_output_response(
                            port_name,
                            _RunValue(value=value),
                        )
                    )
            if materialization.outputs and not accessible_outputs:
                continue
            node_runs.append(
                RunNodeResponse(
                    node_id=materialization.node_id,
                    status="succeeded",
                    error=None,
                    outputs=accessible_outputs,
                )
            )
        return GraphMaterializationsResponse(
            graph_id=graph_id,
            graph_revision=graph_revision,
            node_runs=node_runs,
        )

    async def _saved_graph_for_context(
        self,
        graph_id: UUID,
        graph_revision: int,
    ) -> SavedGraphRevision:
        if self._saved_graphs is None:
            raise WorkbenchGraphError(
                "Saved graph context is not configured for this workbench"
            )
        return await self._saved_graphs.get_revision(graph_id, graph_revision)

    async def _validate_materialized_pins(
        self,
        graph_id: UUID,
        graph_revision: int,
        submitted_pins: dict[
            tuple[str, str],
            ArtifactRef | ArtifactRefSequence,
        ],
    ) -> None:
        if not submitted_pins:
            return
        async with self._uow as uow:
            materializations = await uow.materialized_outputs.list_for_graph(
                graph_id,
                graph_revision,
            )
        by_node = {
            materialization.node_id: materialization
            for materialization in materializations
        }

        for (from_node, from_port), submitted_value in submitted_pins.items():
            materialization = by_node.get(from_node)
            materialized_value = (
                materialization.outputs.get(from_port)
                if materialization is not None
                else None
            )
            if materialized_value is None or not await self._run_value_is_accessible(
                materialized_value
            ):
                raise WorkbenchGraphError(
                    f"Cannot reuse upstream output {from_node!r}.{from_port!r}: "
                    "there is no accessible materialized artifact for this graph "
                    "revision. Run the upstream node too or choose "
                    "'Run with dependencies'."
                )
            if submitted_value != materialized_value:
                raise WorkbenchGraphError(
                    f"Pinned output {from_node!r}.{from_port!r} is not the latest "
                    f"materialized output for graph {graph_id} revision "
                    f"{graph_revision}. Refresh the graph materializations and "
                    "try again, or choose 'Run with dependencies'."
                )

    async def _run_value_is_accessible(
        self,
        value: ArtifactRef | ArtifactRefSequence,
    ) -> bool:
        refs = value.item_refs if isinstance(value, ArtifactRefSequence) else [value]
        for ref in refs:
            artifact = await self.get_artifact(ref.artifact_id)
            if artifact is None or artifact.ref() != ref:
                return False
            if artifact.inline_payload is not None:
                continue
            if artifact.bucket is None or artifact.object_key is None:
                return False
            if not self._storage.exists(artifact.bucket, artifact.object_key):
                return False
        return True

    async def _resolve_pinned_outputs(
        self,
        pinned_outputs: dict[
            tuple[str, str],
            ArtifactRef | ArtifactRefSequence,
        ],
    ) -> dict[str, dict[str, _RunValue]]:
        outputs: dict[str, dict[str, _RunValue]] = {}
        for (from_node, from_port), value in pinned_outputs.items():
            refs = (
                value.item_refs if isinstance(value, ArtifactRefSequence) else [value]
            )
            for index, ref in enumerate(refs):
                item_context = (
                    f" sequence item {index}"
                    if isinstance(value, ArtifactRefSequence)
                    else ""
                )
                artifact = await self.get_artifact(ref.artifact_id)
                if artifact is None:
                    raise WorkbenchGraphError(
                        f"Pinned output {from_node!r}.{from_port!r}{item_context} "
                        f"references missing artifact {ref.artifact_id}"
                    )
                if artifact.ref() != ref:
                    raise WorkbenchGraphError(
                        f"Pinned output {from_node!r}.{from_port!r}{item_context} "
                        f"does not match the repository ref for artifact "
                        f"{ref.artifact_id}"
                    )
            if not await self._run_value_is_accessible(value):
                raise WorkbenchGraphError(
                    f"Pinned output {from_node!r}.{from_port!r} is not accessible"
                )
            outputs.setdefault(from_node, {})[from_port] = _RunValue(value=value)
        return outputs

    async def _assemble_inputs(
        self,
        node: Node[Any, Any, Any],
        node_request: RunNodeRequest,
        edges: list[RunEdgeRequest],
        outputs: dict[str, dict[str, _RunValue]],
        run_id: UUID,
    ) -> dict[str, object]:
        values: dict[str, object] = {}
        for name, spec in node.input_contract.ports.items():
            incoming_by_plug = {
                edge.to_plug: edge
                for edge in edges
                if edge.to_node == node_request.id
                and edge.to_port == name
                and edge.to_plug is not None
            }
            if spec.instance_plugs:
                incoming = [
                    incoming_by_plug[plug.id]
                    for plug in node_request.input_plugs
                    if plug.port == name
                ]
            else:
                incoming = [
                    edge
                    for edge in edges
                    if edge.to_node == node_request.id and edge.to_port == name
                ]
            port_values: list[ArtifactRef | ArtifactRefSequence] = []
            for edge in incoming:
                source_ports = outputs.get(edge.from_node)
                if source_ports is None or edge.from_port not in source_ports:
                    raise WorkbenchGraphError(
                        f"Node {node_request.id!r} input {name!r} references "
                        f"missing output {edge.from_node!r}.{edge.from_port!r}"
                    )
                run_value = source_ports[edge.from_port]
                if edge.projection is not None:
                    projection = _field_projection_for(
                        self._projectable_artifact_types,
                        _run_value_key(run_value),
                        tuple(edge.projection.path),
                    )
                    if projection is None:
                        path = ".".join(edge.projection.path)
                        raise WorkbenchGraphError(
                            f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                            f"{edge.to_node!r}.{edge.to_port!r} uses undeclared "
                            f"projection {path!r}"
                        )
                    run_value = await self._project_run_value(
                        run_value,
                        projection,
                        edge,
                        run_id,
                    )
                if edge.conversion_path:
                    conversions = tuple(
                        self._artifact_conversions[
                            ArtifactConversionKey(
                                id=conversion.id,
                                version=conversion.version,
                            )
                        ]
                        for conversion in edge.conversion_path
                    )
                    run_value = await self._convert_run_value(
                        run_value,
                        conversions,
                        edge,
                        run_id,
                    )
                port_values.append(run_value.value)

            if spec.instance_plugs or spec.variadic:
                values[name] = port_values
            elif len(port_values) == 1:
                values[name] = port_values[0]
            elif len(port_values) > 1:
                raise WorkbenchGraphError(
                    f"Node {node_request.id!r} input {name!r} accepts one "
                    f"connection, got {len(port_values)}"
                )
        return values

    async def _convert_run_value(
        self,
        run_value: _RunValue,
        conversions: tuple[ArtifactConversion[Any, Any], ...],
        edge: RunEdgeRequest,
        run_id: UUID,
    ) -> _RunValue:
        final_conversion = conversions[-1]
        if isinstance(run_value.value, ArtifactRef):
            return _RunValue(
                value=await self._convert_ref(
                    run_value.value,
                    conversions,
                    edge,
                    run_id,
                    item_index=None,
                )
            )

        source_sequence = run_value.value
        converted_refs = [
            await self._convert_ref(
                ref,
                conversions,
                edge,
                run_id,
                item_index=index,
            )
            for index, ref in enumerate(source_sequence.item_refs)
        ]
        sequence_metadata = dict(source_sequence.metadata)
        sequence_metadata.update(
            {
                "source_sequence_id": str(source_sequence.sequence_id),
                **_conversion_path_metadata(conversions),
            }
        )
        return _RunValue(
            value=ArtifactRefSequence(
                artifact_type=final_conversion.target.id,
                schema_version=final_conversion.target.schema_version,
                item_refs=converted_refs,
                ordered=source_sequence.ordered,
                index_key=source_sequence.index_key,
                metadata=sequence_metadata,
            )
        )

    async def _convert_ref(
        self,
        ref: ArtifactRef,
        conversions: tuple[ArtifactConversion[Any, Any], ...],
        edge: RunEdgeRequest,
        run_id: UUID,
        *,
        item_index: int | None,
    ) -> ArtifactRef:
        first_conversion = conversions[0]
        final_conversion = conversions[-1]
        item_context = "" if item_index is None else f" at sequence item {item_index}"
        try:
            source_value = await self._resolvers.resolve(
                ref=ref,
                target=first_conversion.source_type,
            )
            converted_value: object = _validated_conversion_value(
                source_value,
                first_conversion.source_type,
            )
        except Exception as exc:
            message = (
                f"Failed to resolve artifact {ref.artifact_id}{item_context} for "
                f"conversion path on edge "
                f"{edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r}"
            )
            raise WorkbenchGraphError(message) from exc

        for step_index, conversion in enumerate(conversions):
            try:
                step_input = _validated_conversion_value(
                    converted_value,
                    conversion.source_type,
                )
                converted_value = _validated_conversion_value(
                    conversion.convert(step_input),
                    conversion.target_type,
                )
            except Exception as exc:
                message = (
                    f"Failed conversion step {step_index + 1}/{len(conversions)} "
                    f"{conversion.key.id!r}@{conversion.key.version} for artifact "
                    f"{ref.artifact_id}{item_context} on edge "
                    f"{edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r}"
                )
                raise WorkbenchGraphError(message) from exc

        metadata: dict[str, object] = {
            "source_artifact_id": str(ref.artifact_id),
            "source_artifact_type": ref.artifact_type,
            "source_schema_version": ref.schema_version,
            "source_node_id": edge.from_node,
            "source_port": edge.from_port,
            **_conversion_path_metadata(conversions),
            "conversion_source_artifact_type": first_conversion.source.id,
            "conversion_source_schema_version": first_conversion.source.schema_version,
            "conversion_target_artifact_type": final_conversion.target.id,
            "conversion_target_schema_version": final_conversion.target.schema_version,
            "target_node_id": edge.to_node,
            "target_port": edge.to_port,
        }
        try:
            writer = self._writers.writer_for(final_conversion.target)
            written_ref = await writer.write(
                converted_value,
                ArtifactWriteContext(
                    node_context=NodeExecutionContext(
                        workflow_run_id=run_id,
                        node_run_id=uuid4(),
                        node_id=edge.from_node,
                    ),
                    provenance=MaterializationProvenance(
                        refs_by_input={edge.from_port: (ref,)}
                    ),
                    item_index=item_index,
                    metadata=metadata,
                ),
            )
            if written_ref.key() != final_conversion.target:
                raise WorkbenchGraphError(
                    f"Final conversion writer returned {written_ref.artifact_type}@"
                    f"{written_ref.schema_version}, expected "
                    f"{final_conversion.target.id}@"
                    f"{final_conversion.target.schema_version} for artifact "
                    f"{ref.artifact_id}{item_context} on edge "
                    f"{edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r}"
                )
            return written_ref
        except WorkbenchGraphError:
            raise
        except Exception as exc:
            message = (
                f"Failed to materialize final target "
                f"{final_conversion.target.id}@"
                f"{final_conversion.target.schema_version} for conversion path "
                f"from artifact {ref.artifact_id}{item_context} on edge "
                f"{edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r}"
            )
            raise WorkbenchGraphError(message) from exc

    async def _project_run_value(
        self,
        run_value: _RunValue,
        projection: ArtifactFieldProjection,
        edge: RunEdgeRequest,
        run_id: UUID,
    ) -> _RunValue:
        if isinstance(run_value.value, ArtifactRef):
            return _RunValue(
                value=await self._project_ref(
                    run_value.value,
                    projection,
                    edge,
                    run_id,
                    item_index=None,
                )
            )

        source_sequence = run_value.value
        projected_refs = [
            await self._project_ref(
                ref,
                projection,
                edge,
                run_id,
                item_index=index,
            )
            for index, ref in enumerate(source_sequence.item_refs)
        ]
        sequence_metadata = dict(source_sequence.metadata)
        sequence_metadata.update(
            {
                "source_sequence_id": str(source_sequence.sequence_id),
                "projection_path": list(projection.path),
                "projection_title": projection.title,
            }
        )
        return _RunValue(
            value=ArtifactRefSequence(
                artifact_type=projection.target.id,
                schema_version=projection.target.schema_version,
                item_refs=projected_refs,
                ordered=source_sequence.ordered,
                index_key=source_sequence.index_key,
                metadata=sequence_metadata,
            )
        )

    async def _project_ref(
        self,
        ref: ArtifactRef,
        projection: ArtifactFieldProjection,
        edge: RunEdgeRequest,
        run_id: UUID,
        *,
        item_index: int | None,
    ) -> ArtifactRef:
        artifact = await self.get_artifact(ref.artifact_id)
        if artifact is None:
            raise WorkbenchGraphError(
                f"Cannot project missing source artifact {ref.artifact_id} for "
                f"edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r}"
            )
        if artifact.ref() != ref:
            raise WorkbenchGraphError(
                f"Cannot project source artifact {ref.artifact_id}: repository "
                f"ref does not match edge output ref"
            )
        if artifact.inline_payload is None:
            raise WorkbenchGraphError(
                f"Cannot project {'.'.join(projection.path)!r} from artifact "
                f"{ref.artifact_id}: source has no inline JSON payload"
            )

        projected_value: object = artifact.inline_payload
        for segment in projection.path:
            if not isinstance(projected_value, dict):
                raise WorkbenchGraphError(
                    f"Cannot project {'.'.join(projection.path)!r} from artifact "
                    f"{ref.artifact_id}: {segment!r} is not inside a JSON object"
                )
            mapping = cast(dict[object, object], projected_value)
            if segment not in mapping:
                raise WorkbenchGraphError(
                    f"Cannot project {'.'.join(projection.path)!r} from artifact "
                    f"{ref.artifact_id}: field {segment!r} is missing"
                )
            projected_value = mapping[segment]

        writer = self._writers.writer_for(projection.target)
        return await writer.write(
            projected_value,
            ArtifactWriteContext(
                node_context=NodeExecutionContext(
                    workflow_run_id=run_id,
                    node_run_id=uuid4(),
                    node_id=edge.from_node,
                ),
                provenance=MaterializationProvenance(
                    refs_by_input={edge.from_port: (ref,)}
                ),
                item_index=item_index,
                metadata={
                    "source_artifact_id": str(ref.artifact_id),
                    "source_artifact_type": ref.artifact_type,
                    "source_schema_version": ref.schema_version,
                    "source_node_id": edge.from_node,
                    "source_port": edge.from_port,
                    "projection_path": list(projection.path),
                    "projection_title": projection.title,
                    "projection_target_artifact_type": projection.target.id,
                    "projection_target_schema_version": (
                        projection.target.schema_version
                    ),
                    "target_node_id": edge.to_node,
                    "target_port": edge.to_port,
                },
            ),
        )

    async def _port_output_response(
        self,
        port_name: str,
        run_value: _RunValue,
    ) -> RunPortOutputResponse:
        if isinstance(run_value.value, ArtifactRefSequence):
            refs = list(run_value.value.item_refs)
            kind: Literal["single", "sequence"] = "sequence"
        else:
            refs = [run_value.value]
            kind = "single"
        return RunPortOutputResponse(
            port=port_name,
            kind=kind,
            value=run_value.value,
            artifacts=[await self._artifact_summary(ref) for ref in refs],
        )

    async def _artifact_summary(
        self,
        ref: ArtifactRef,
    ) -> ArtifactSummaryResponse:
        artifact = await self.get_artifact(ref.artifact_id)
        if artifact is None:
            return ArtifactSummaryResponse(
                artifact_id=ref.artifact_id,
                artifact_type=ref.artifact_type,
                schema_version=ref.schema_version,
                content_type="application/octet-stream",
            )
        text: str | None = None
        if artifact.inline_payload is not None:
            payload_text = artifact.inline_payload.get("text")
            if isinstance(payload_text, str):
                text = payload_text
            payload_markdown = artifact.inline_payload.get("markdown")
            if text is None and isinstance(payload_markdown, str):
                text = payload_markdown
            if text is None:
                if set(artifact.inline_payload) == {"value"}:
                    text = json.dumps(
                        artifact.inline_payload["value"],
                        ensure_ascii=False,
                    )
                else:
                    text = json.dumps(
                        artifact.inline_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
        content_url = f"./artifacts/{artifact.id}/content"
        return ArtifactSummaryResponse(
            artifact_id=artifact.id,
            artifact_type=artifact.artifact_type,
            schema_version=artifact.schema_version,
            content_type=artifact.content_type,
            byte_size=artifact.byte_size,
            sha256=artifact.sha256,
            text=text,
            content_url=content_url,
            metadata=artifact.metadata,
        )

    async def get_artifact(self, artifact_id: UUID) -> ArtifactObject | None:
        async with self._uow as uow:
            return await uow.artifacts.get(artifact_id)

    async def load_artifact_content(self, artifact: ArtifactObject) -> bytes:
        if artifact.inline_payload is not None:
            return (
                json.dumps(
                    artifact.inline_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
        if artifact.bucket is None or artifact.object_key is None:
            raise WorkbenchGraphError(f"Artifact {artifact.id} has no stored payload")
        stream = await self._storage.load(
            bucket=artifact.bucket,
            path=artifact.object_key,
        )
        try:
            return stream.read()
        finally:
            stream.close()


def _validate_saved_graph_fragment(
    graph: SavedGraph | SavedGraphRevision,
    nodes: list[RunNodeRequest],
    edges: list[RunEdgeRequest],
) -> None:
    saved_nodes = {node.id: node for node in graph.document.nodes}
    for node in nodes:
        saved_node = saved_nodes.get(node.id)
        if saved_node is None:
            raise WorkbenchGraphError(
                f"Run node {node.id!r} does not belong to saved graph {graph.id} "
                f"revision {graph.revision}"
            )
        if (
            node.operator_id != saved_node.operator_id
            or node.operator_version != saved_node.operator_version
            or node.config != saved_node.config_dict()
            or tuple((plug.id, plug.port) for plug in node.input_plugs)
            != tuple((plug.id, plug.port) for plug in saved_node.input_plugs)
            or {
                binding.variable: binding.artifact_type.to_key()
                for binding in node.artifact_type_bindings
            }
            != {
                binding.variable: binding.artifact_type
                for binding in saved_node.artifact_type_bindings
            }
        ):
            raise WorkbenchGraphError(
                f"Run node {node.id!r} does not match saved graph {graph.id} "
                f"revision {graph.revision}"
            )

    executed_node_ids = {node.id for node in nodes}
    saved_incoming_edges = Counter(
        (
            edge.from_node,
            edge.from_port,
            edge.to_node,
            edge.to_port,
            edge.to_plug,
            edge.collection_mode,
            tuple(edge.projection.path) if edge.projection is not None else None,
            tuple(
                (conversion.id, conversion.version)
                for conversion in edge.conversion_path
            ),
        )
        for edge in graph.document.edges
        if edge.to_node in executed_node_ids
    )
    submitted_edges = Counter(
        (
            edge.from_node,
            edge.from_port,
            edge.to_node,
            edge.to_port,
            edge.to_plug,
            edge.collection_mode,
            tuple(edge.projection.path) if edge.projection is not None else None,
            tuple(
                (conversion.id, conversion.version)
                for conversion in edge.conversion_path
            ),
        )
        for edge in edges
    )
    if submitted_edges != saved_incoming_edges:
        missing_count = sum((saved_incoming_edges - submitted_edges).values())
        unexpected_count = sum((submitted_edges - saved_incoming_edges).values())
        raise WorkbenchGraphError(
            "Run edges do not match the saved incoming edges for the executed "
            f"nodes in graph {graph.id} revision {graph.revision}: "
            f"{missing_count} missing and {unexpected_count} unexpected or duplicated"
        )


def _validate_secret_graph_bindings(
    graph: SavedGraph | SavedGraphRevision,
    nodes: list[RunNodeRequest],
    plugin_registry: PluginRegistry,
) -> set[str]:
    registrations = {
        registration.key: registration for registration in plugin_registry.nodes
    }
    saved_nodes = {node.id: node for node in graph.document.nodes}
    validated_node_ids: set[str] = set()

    for node in nodes:
        registration = registrations.get((node.operator_id, node.operator_version))
        if registration is None or not registration.secret_inputs:
            continue
        saved_node = saved_nodes.get(node.id)
        if saved_node is None:
            raise WorkbenchGraphError(
                f"Secret-bearing run node {node.id!r} does not belong to saved "
                f"graph {graph.id} revision {graph.revision}"
            )
        if (
            saved_node.operator_id != node.operator_id
            or saved_node.operator_version != node.operator_version
        ):
            raise WorkbenchGraphError(
                f"Secret-bearing run node {node.id!r} does not match the saved "
                f"operator in graph {graph.id} revision {graph.revision}"
            )

        config_model = registration.node_class.config_contract.model
        try:
            submitted_config = config_model.model_validate(node.config).model_dump(
                mode="json"
            )
            saved_config = config_model.model_validate(
                saved_node.config_dict()
            ).model_dump(mode="json")
        except ValueError as exc:
            raise WorkbenchGraphError(
                f"Secret-bearing run node {node.id!r} has invalid configuration"
            ) from exc

        for declaration in registration.secret_inputs:
            submitted_dependencies = {
                dependency: cast(JsonValue, submitted_config[dependency])
                for dependency in declaration.config_dependencies
            }
            saved_dependencies = {
                dependency: cast(JsonValue, saved_config[dependency])
                for dependency in declaration.config_dependencies
            }
            try:
                dependencies_match = canonical_node_secret_dependencies(
                    submitted_dependencies
                ) == canonical_node_secret_dependencies(saved_dependencies)
            except InvalidNodeSecretDependenciesError as exc:
                raise WorkbenchGraphError(
                    f"Secret-bearing run node {node.id!r} has invalid secret "
                    "configuration dependencies"
                ) from exc
            if not dependencies_match:
                raise WorkbenchGraphError(
                    f"Secret-bearing run node {node.id!r} does not match the "
                    f"saved configuration required by secret input "
                    f"{declaration.name!r}"
                )
        validated_node_ids.add(node.id)

    return validated_node_ids


def _topological_order(
    nodes: list[RunNodeRequest],
    edges: list[RunEdgeRequest],
) -> list[RunNodeRequest]:
    by_id = {node.id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise WorkbenchGraphError("Duplicate node ids in graph")
    for edge in edges:
        if edge.to_node not in by_id:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} references unknown target "
                f"node {edge.to_node!r}"
            )

    incoming_count = {node.id: 0 for node in nodes}
    for edge in edges:
        if edge.from_node in by_id:
            incoming_count[edge.to_node] += 1

    queue = deque(node for node in nodes if incoming_count[node.id] == 0)
    ordered: list[RunNodeRequest] = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for edge in edges:
            if edge.from_node != node.id or edge.to_node not in by_id:
                continue
            incoming_count[edge.to_node] -= 1
            if incoming_count[edge.to_node] == 0:
                queue.append(by_id[edge.to_node])

    if len(ordered) != len(nodes):
        raise WorkbenchGraphError("Graph contains a cycle")
    return ordered


def _pinned_outputs_by_endpoint(
    nodes: list[RunNodeRequest],
    edges: list[RunEdgeRequest],
    pinned_outputs: list[PinnedOutputRequest],
) -> dict[tuple[str, str], ArtifactRef | ArtifactRefSequence]:
    node_ids = {node.id for node in nodes}
    by_endpoint: dict[tuple[str, str], ArtifactRef | ArtifactRefSequence] = {}
    for pinned_output in pinned_outputs:
        endpoint = (pinned_output.from_node, pinned_output.from_port)
        if endpoint in by_endpoint:
            raise WorkbenchGraphError(
                f"Duplicate pinned output for {pinned_output.from_node!r}."
                f"{pinned_output.from_port!r}"
            )
        if pinned_output.from_node in node_ids:
            raise WorkbenchGraphError(
                f"Pinned output {pinned_output.from_node!r}."
                f"{pinned_output.from_port!r} is invalid because source node "
                f"{pinned_output.from_node!r} is also being executed"
            )
        by_endpoint[endpoint] = pinned_output.value

    external_endpoints: set[tuple[str, str]] = set()
    for edge in edges:
        if edge.from_node in node_ids:
            continue
        endpoint = (edge.from_node, edge.from_port)
        external_endpoints.add(endpoint)
        if endpoint not in by_endpoint:
            raise WorkbenchGraphError(
                f"External edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} requires a pinned output"
            )

    for from_node, from_port in by_endpoint:
        if (from_node, from_port) not in external_endpoints:
            raise WorkbenchGraphError(
                f"Pinned output {from_node!r}.{from_port!r} is not used by any "
                "incoming edge"
            )
    return by_endpoint


def _validate_input_plugs(
    nodes_by_id: dict[str, Node[Any, Any, Any]],
    node_requests: list[RunNodeRequest],
    edges: list[RunEdgeRequest],
) -> None:
    plugs_by_node: dict[str, dict[str, str]] = {}
    for node_request in node_requests:
        node = nodes_by_id[node_request.id]
        plugs: dict[str, str] = {}
        for plug in node_request.input_plugs:
            if plug.id in plugs:
                raise WorkbenchGraphError(
                    f"Node {node_request.id!r} has duplicate input plug id {plug.id!r}"
                )
            target_port = node.input_contract.ports.get(plug.port)
            if target_port is None:
                raise WorkbenchGraphError(
                    f"Node {node_request.id!r} input plug {plug.id!r} references "
                    f"unknown input port {plug.port!r}"
                )
            if not target_port.instance_plugs:
                raise WorkbenchGraphError(
                    f"Node {node_request.id!r} input port {plug.port!r} does not "
                    "accept instance plugs"
                )
            plugs[plug.id] = plug.port
        plugs_by_node[node_request.id] = plugs

        for port_name, port in node.input_contract.ports.items():
            if not port.instance_plugs or not port.required:
                continue
            if any(plug.port == port_name for plug in node_request.input_plugs):
                continue
            raise WorkbenchGraphError(
                f"Node {node_request.id!r} ({node.operator_id}@"
                f"{node.operator_version}) required instance-plug input "
                f"{port_name!r} has no submitted plugs"
            )

    incoming_by_plug: Counter[tuple[str, str]] = Counter()
    for edge in edges:
        target_node = nodes_by_id[edge.to_node]
        target_port = target_node.input_contract.ports.get(edge.to_port)
        if target_port is None:
            continue
        if not target_port.instance_plugs:
            if edge.to_plug is not None:
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} declares input plug "
                    f"{edge.to_plug!r}, but the target port does not accept "
                    "instance plugs"
                )
            continue
        if edge.collection_mode == "map":
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} cannot use collection mode "
                "'map' with an instance-plug input"
            )
        if edge.to_plug is None:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} must target an input plug"
            )
        plug_port = plugs_by_node[edge.to_node].get(edge.to_plug)
        if plug_port is None:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} targets unknown input plug "
                f"{edge.to_plug!r}"
            )
        if plug_port != edge.to_port:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} targets input plug "
                f"{edge.to_plug!r}, which belongs to port {plug_port!r}"
            )
        plug_key = (edge.to_node, edge.to_plug)
        incoming_by_plug[plug_key] += 1
        if incoming_by_plug[plug_key] > 1:
            raise WorkbenchGraphError(
                f"Node {edge.to_node!r} input plug {edge.to_plug!r} requires "
                "exactly one incoming edge"
            )

    for node_id, plugs in plugs_by_node.items():
        for plug_id in plugs:
            if incoming_by_plug[(node_id, plug_id)] == 1:
                continue
            raise WorkbenchGraphError(
                f"Node {node_id!r} input plug {plug_id!r} requires exactly one "
                "incoming edge"
            )


def _validate_edges(
    nodes_by_id: dict[str, Node[Any, Any, Any]],
    resolved_contracts_by_node: dict[str, ResolvedNodeContracts],
    invocations_by_id: dict[str, NodeInvocation],
    edges: list[RunEdgeRequest],
    projectable_artifact_types: dict[ArtifactTypeKey, ArtifactTypeSpec],
    artifact_conversions: dict[
        ArtifactConversionKey,
        ArtifactConversion[Any, Any],
    ],
    pinned_outputs: dict[
        tuple[str, str],
        ArtifactRef | ArtifactRefSequence,
    ],
) -> None:
    incoming_counts: dict[tuple[str, str], int] = {}
    for edge in edges:
        target_node = nodes_by_id[edge.to_node]
        target_port = resolved_contracts_by_node[edge.to_node].input_contract.ports.get(
            edge.to_port
        )
        if target_port is None:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} references unknown input "
                f"port {edge.to_port!r} on node {edge.to_node!r}"
            )
        target_key = target_port.accepts
        if not isinstance(target_key, ArtifactTypeKey):
            raise WorkbenchGraphError(
                f"Node {edge.to_node!r} input {edge.to_port!r} retained "
                f"unresolved artifact type variable {target_key.name!r}"
            )
        source_node = nodes_by_id.get(edge.from_node)
        if source_node is None:
            pinned_value = pinned_outputs[(edge.from_node, edge.from_port)]
            source_shape = (
                PortShape.MANY
                if isinstance(pinned_value, ArtifactRefSequence)
                else PortShape.ONE
            )
            source_key = _value_key(pinned_value)
        else:
            source_port = resolved_contracts_by_node[
                edge.from_node
            ].output_contract.ports.get(edge.from_port)
            if source_port is None:
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} references unknown output "
                    f"port {edge.from_port!r} on node {edge.from_node!r}"
                )
            source_shape = effective_output_shape(
                source_node,
                invocations_by_id[edge.from_node],
                edge.from_port,
            )
            source_key = source_port.produces
            if not isinstance(source_key, ArtifactTypeKey):
                raise WorkbenchGraphError(
                    f"Node {edge.from_node!r} output {edge.from_port!r} retained "
                    f"unresolved artifact type variable {source_key.name!r}"
                )
        if edge.collection_mode == "map" and source_shape is not PortShape.MANY:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} uses collection mode 'map', "
                f"which requires a source with shape {PortShape.MANY.value!r}; "
                f"source is {source_shape.value!r}"
            )
        incoming_key = (edge.to_node, edge.to_port)
        incoming_counts[incoming_key] = incoming_counts.get(incoming_key, 0) + 1
        if not target_port.variadic and incoming_counts[incoming_key] > 1:
            raise WorkbenchGraphError(
                f"Node {edge.to_node!r} input {edge.to_port!r} accepts one "
                f"connection, got {incoming_counts[incoming_key]}"
            )

        effective_source_key = source_key
        if edge.projection is not None:
            requested_path = tuple(edge.projection.path)
            projection = _field_projection_for(
                projectable_artifact_types,
                effective_source_key,
                requested_path,
            )
            if projection is None:
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} requests undeclared "
                    f"projection {'.'.join(requested_path)!r} on "
                    f"{effective_source_key.id}@"
                    f"{effective_source_key.schema_version}"
                )
            effective_source_key = projection.target

        if len(edge.conversion_path) > MAX_ARTIFACT_CONVERSION_HOPS:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} conversion path exceeds "
                f"the maximum of {MAX_ARTIFACT_CONVERSION_HOPS} steps"
            )
        seen_artifact_keys = {effective_source_key}
        resolved_conversions: list[ArtifactConversion[Any, Any]] = []
        for step_index, requested_conversion in enumerate(edge.conversion_path):
            conversion_key = ArtifactConversionKey(
                id=requested_conversion.id,
                version=requested_conversion.version,
            )
            conversion = artifact_conversions.get(conversion_key)
            if conversion is None:
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} requests undeclared "
                    f"conversion {conversion_key.id!r}@{conversion_key.version} "
                    f"at step {step_index + 1}"
                )
            if conversion.source != effective_source_key:
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} applies conversion step "
                    f"{step_index + 1} "
                    f"{conversion.key.id!r}@{conversion.key.version}, which expects "
                    f"{conversion.source.id}@{conversion.source.schema_version}, "
                    f"to {effective_source_key.id}@"
                    f"{effective_source_key.schema_version}"
                )
            if resolved_conversions and not conversion_runtime_types_are_compatible(
                resolved_conversions[-1].target_type,
                conversion.source_type,
            ):
                previous = resolved_conversions[-1]
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} conversion steps "
                    f"{step_index} and {step_index + 1} have incompatible runtime "
                    f"types: {previous.target_type} does not match "
                    f"{conversion.source_type}"
                )
            if conversion.target in seen_artifact_keys:
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} conversion path repeats "
                    f"artifact type {conversion.target.id}@"
                    f"{conversion.target.schema_version} at step {step_index + 1}"
                )
            resolved_conversions.append(conversion)
            effective_source_key = conversion.target
            seen_artifact_keys.add(effective_source_key)

        if effective_source_key != target_key:
            if resolved_conversions:
                conversion_path = " -> ".join(
                    f"{conversion.key.id}@{conversion.key.version}"
                    for conversion in resolved_conversions
                )
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} converts through "
                    f"{conversion_path} as "
                    f"{effective_source_key.id}@"
                    f"{effective_source_key.schema_version}, but target expects "
                    f"{target_key.id}@"
                    f"{target_key.schema_version}"
                )
            if edge.projection is not None:
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} projects "
                    f"{'.'.join(edge.projection.path)!r} as "
                    f"{effective_source_key.id}@"
                    f"{effective_source_key.schema_version}, but target expects "
                    f"{target_key.id}@"
                    f"{target_key.schema_version}"
                )
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} cannot connect "
                f"{effective_source_key.id}@"
                f"{effective_source_key.schema_version} to "
                f"{target_key.id}@{target_key.schema_version} "
                "without a declared field projection or conversion"
            )

        invocation = invocations_by_id[edge.to_node]
        if (
            invocation.mode is InvocationMode.MAP
            and invocation.map_input == edge.to_port
        ):
            accepted_shapes = (
                effective_input_shape(
                    target_node,
                    invocation,
                    edge.to_port,
                ),
            )
        else:
            accepted_shapes = target_port.accepted_shapes
        if source_shape in accepted_shapes:
            continue
        expected_shapes = ", ".join(repr(shape.value) for shape in accepted_shapes)
        if len(accepted_shapes) == 1:
            target_shapes = f"expects {expected_shapes}"
        else:
            target_shapes = f"accepts one of {expected_shapes}"
        raise WorkbenchGraphError(
            f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
            f"{edge.to_node!r}.{edge.to_port!r} has incompatible shapes: "
            f"source is {source_shape.value!r}, target {target_shapes}"
        )

    for node_id, node in nodes_by_id.items():
        for port_name, port in node.input_contract.ports.items():
            if not port.required:
                continue
            if incoming_counts.get((node_id, port_name), 0) > 0:
                continue
            raise WorkbenchGraphError(
                f"Node {node_id!r} ({node.operator_id}@"
                f"{node.operator_version}) required input {port_name!r} has no "
                f"incoming edge"
            )


def _derive_invocations(
    nodes_by_id: dict[str, Node[Any, Any, Any]],
    edges: list[RunEdgeRequest],
) -> dict[str, NodeInvocation]:
    map_edges_by_target: dict[str, RunEdgeRequest] = {}
    for edge in edges:
        if edge.collection_mode != "map":
            continue
        if edge.to_port.strip() == "":
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} cannot drive mapped "
                "execution without a target port"
            )
        existing = map_edges_by_target.get(edge.to_node)
        if existing is not None:
            raise WorkbenchGraphError(
                f"Node {edge.to_node!r} has more than one map edge: "
                f"{existing.from_node!r}.{existing.from_port!r} -> "
                f"{existing.to_port!r} and {edge.from_node!r}.{edge.from_port!r} "
                f"-> {edge.to_port!r}; exactly one edge may drive mapped "
                "execution"
            )
        map_edges_by_target[edge.to_node] = edge

    invocations: dict[str, NodeInvocation] = {}
    for node_id, node in nodes_by_id.items():
        map_edge = map_edges_by_target.get(node_id)
        if map_edge is None:
            invocations[node_id] = NodeInvocation()
            continue

        invocation = NodeInvocation(
            mode=InvocationMode.MAP,
            map_input=map_edge.to_port,
        )
        try:
            validate_invocation(node, invocation)
        except InvocationError as exc:
            raise WorkbenchGraphError(
                f"Edge {map_edge.from_node!r}.{map_edge.from_port!r} -> "
                f"{map_edge.to_node!r}.{map_edge.to_port!r} cannot drive "
                f"mapped execution: {exc}"
            ) from exc
        invocations[node_id] = invocation
    return invocations


def _conversion_path_metadata(
    conversions: tuple[ArtifactConversion[Any, Any], ...],
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "conversion_path": [
            {
                "id": conversion.key.id,
                "version": conversion.key.version,
            }
            for conversion in conversions
        ],
        "conversion_titles": [conversion.title for conversion in conversions],
    }
    if len(conversions) == 1:
        conversion = conversions[0]
        metadata.update(
            {
                "conversion_id": conversion.key.id,
                "conversion_version": conversion.key.version,
                "conversion_title": conversion.title,
            }
        )
    return metadata


def _validated_conversion_value[ValueT](
    value: object,
    target: type[ValueT],
) -> ValueT:
    if issubclass(target, BaseModel):
        if not isinstance(value, target):
            raise TypeError(
                f"Expected {target.__module__}.{target.__qualname__}, got "
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        raw_value = value.model_dump(mode="python", round_trip=True)
        return target.model_validate(raw_value, strict=True)
    return TypeAdapter(
        target,
        config=ConfigDict(arbitrary_types_allowed=True),
    ).validate_python(value, strict=True)


def _field_projection_for(
    projectable_artifact_types: dict[ArtifactTypeKey, ArtifactTypeSpec],
    artifact_type: ArtifactTypeKey,
    path: tuple[str, ...],
) -> ArtifactFieldProjection | None:
    artifact_spec = projectable_artifact_types.get(artifact_type)
    if artifact_spec is None:
        return None
    for projection in artifact_spec.field_projections:
        if projection.path == path:
            return projection
    return None


def _run_value_key(run_value: _RunValue) -> ArtifactTypeKey:
    return _value_key(run_value.value)


def _value_key(value: ArtifactRef | ArtifactRefSequence) -> ArtifactTypeKey:
    if isinstance(value, ArtifactRef):
        return value.key()
    return ArtifactTypeKey(
        value.artifact_type,
        value.schema_version,
    )


def _port_values(
    node: Node[Any, Any, Any],
    result: object,
) -> dict[str, _RunValue]:
    values: dict[str, _RunValue] = {}
    if not isinstance(result, PersistedNodeOutput):
        return values
    for name in node.output_contract.ports:
        value = result.values.get(name)
        if isinstance(value, ArtifactRef | ArtifactRefSequence):
            values[name] = _RunValue(value=value)
    return values
