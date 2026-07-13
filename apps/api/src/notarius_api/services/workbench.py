"""In-process execution backend for the workbench UI."""

import base64
import binascii
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID, uuid4

from PIL import Image as ImageModule
from PIL import ImageDraw

from notarius_core.artifacts import (
    TABLE_FRAGMENT,
    TABLE_PAGE,
    ArtifactFieldProjection,
    ArtifactObject,
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    ArtifactTypeSpec,
    InMemoryUnitOfWork,
)
from notarius_core.nodes import Node, NodeExecutionContext, PortShape
from notarius_core.operators.arithmetic import (
    ARITHMETIC_RESULT,
    ArithmeticResult,
    IntegerValueOutputWriter,
    IntegerValueResolver,
)
from notarius_core.operators.tables import (
    TableFragment,
    TablePage,
)
from notarius_core.operators.text import TEXT_VALUE, TextValue
from notarius_core.plugins import (
    PluginRegistry,
    PluginRuntimeContext,
    UnknownOperatorError,
)
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
    InlineModelOutputWriter,
    OutputPersister,
    PersistedNodeOutput,
    SourcePageImageOutputWriter,
    TableCsvBundleOutputWriter,
)
from notarius_core.runtime.resolvers import (
    InlineModelResolver,
    Resolver,
    ResolverRegistry,
)
from notarius_storage import LocalFileObjectStore

from notarius_api.schemas.workbench import (
    ArtifactSummaryResponse,
    RunEdgeRequest,
    RunNodeRequest,
    RunNodeResponse,
    RunPortOutputResponse,
    RunRequest,
    RunResponse,
    SelectionItemResponse,
)

_WORKBENCH_BUCKET = "workbench-artifacts"
_SAMPLE_PAGE_TEXTS = (
    "PAGE {index}\nParochia Sancti Floriani\nAnno Domini 1846",
    "PAGE {index}\nBaptisatorum liber\nVilla Nova, folio {index}",
    "PAGE {index}\nIndex nominum\nSeries continua",
)


class WorkbenchGraphError(RuntimeError):
    pass


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


class WorkbenchService:
    def __init__(
        self,
        *,
        plugin_registry: PluginRegistry,
        workspace: Path | None = None,
    ) -> None:
        self._plugin_registry = plugin_registry
        self._workspace = workspace or _default_workspace()
        self._uploads_dir = self._workspace / "uploads"
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._storage = LocalFileObjectStore(self._workspace / "objects")
        self._uow = InMemoryUnitOfWork()
        self._plugin_context = PluginRuntimeContext(
            workspace=self._workspace,
            uploads_dir=self._uploads_dir,
            storage=self._storage,
            uow=self._uow,
            bucket=_WORKBENCH_BUCKET,
        )
        resolvers = [
            cast(Resolver[object], IntegerValueResolver(uow=self._uow)),
            cast(
                Resolver[object],
                InlineModelResolver(
                    source=ARITHMETIC_RESULT.key,
                    target=ArithmeticResult,
                    uow=self._uow,
                ),
            ),
            cast(
                Resolver[object],
                InlineModelResolver(
                    source=TEXT_VALUE.key,
                    target=TextValue,
                    uow=self._uow,
                ),
            ),
            cast(
                Resolver[object],
                InlineModelResolver(
                    source=TABLE_FRAGMENT.key,
                    target=TableFragment,
                    uow=self._uow,
                ),
            ),
            cast(
                Resolver[object],
                InlineModelResolver(
                    source=TABLE_PAGE.key,
                    target=TablePage,
                    uow=self._uow,
                ),
            ),
        ]
        resolvers.extend(plugin_registry.build_resolvers(self._plugin_context))
        self._resolvers = ResolverRegistry(resolvers)

        writers: list[ArtifactOutputWriter] = [
            IntegerValueOutputWriter(uow=self._uow),
            SourcePageImageOutputWriter(
                storage=self._storage,
                uow=self._uow,
                bucket=_WORKBENCH_BUCKET,
            ),
            InlineModelOutputWriter(
                artifact_type=ARITHMETIC_RESULT.key,
                model=ArithmeticResult,
                uow=self._uow,
            ),
            InlineModelOutputWriter(
                artifact_type=TEXT_VALUE.key,
                model=TextValue,
                uow=self._uow,
            ),
            InlineModelOutputWriter(
                artifact_type=TABLE_FRAGMENT.key,
                model=TableFragment,
                uow=self._uow,
            ),
            InlineModelOutputWriter(
                artifact_type=TABLE_PAGE.key,
                model=TablePage,
                uow=self._uow,
            ),
            TableCsvBundleOutputWriter(
                storage=self._storage,
                uow=self._uow,
                bucket=_WORKBENCH_BUCKET,
            ),
        ]
        writers.extend(plugin_registry.build_writers(self._plugin_context))
        self._writers = ArtifactWriterRegistry(writers)
        self._projectable_artifact_types = {
            artifact_type.key: artifact_type
            for artifact_type in plugin_registry.artifact_types
            if artifact_type.field_projections
        }
        self._runtime = NodeRuntime(
            materializer=InputMaterializer(self._resolvers),
            persister=OutputPersister(self._writers),
        )

    @property
    def plugin_registry(self) -> PluginRegistry:
        return self._plugin_registry

    def _build_node(
        self,
        operator_id: str,
        operator_version: int,
    ) -> Node[Any, Any, Any]:
        try:
            return self._plugin_registry.build_node(
                operator_id,
                operator_version,
                self._plugin_context,
            )
        except UnknownOperatorError as exc:
            raise WorkbenchGraphError(str(exc)) from exc

    async def save_upload(
        self,
        filename: str,
        content_base64: str,
    ) -> SelectionItemResponse:
        try:
            content = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WorkbenchGraphError("Upload is not valid base64") from exc

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-") or "upload"
        path = self._uploads_dir / f"{uuid4().hex[:8]}-{safe_name}"
        path.write_bytes(content)
        return self._selection_item(path, display_name=filename)

    async def create_sample_pages(
        self,
        count: int,
    ) -> list[SelectionItemResponse]:
        items: list[SelectionItemResponse] = []
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
                self._selection_item(path, display_name=f"sample-page-{index + 1}.png")
            )
        return items

    def _selection_item(
        self,
        path: Path,
        display_name: str,
    ) -> SelectionItemResponse:
        return SelectionItemResponse(
            connector_id="local_upload",
            external_uri=path.as_uri(),
            display_name=display_name,
            size_bytes=path.stat().st_size,
        )

    async def run_graph(self, request: RunRequest) -> RunResponse:
        order = _topological_order(request.nodes, request.edges)
        nodes_by_id = {
            node_request.id: self._build_node(
                node_request.operator_id,
                node_request.operator_version,
            )
            for node_request in order
        }
        invocations_by_id = _derive_invocations(
            nodes_by_id,
            request.edges,
        )

        _validate_edges(
            nodes_by_id,
            invocations_by_id,
            request.edges,
            self._projectable_artifact_types,
        )
        outputs: dict[str, dict[str, _RunValue]] = {}
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
                result = await self._runtime.run_node(
                    node,
                    NodeExecutionContext(
                        workflow_run_id=run_id,
                        node_run_id=uuid4(),
                        node_id=node_request.id,
                    ),
                    inputs,
                    config=node_request.config,
                    invocation=invocations_by_id[node_request.id],
                )
            except Exception as exc:
                failed.add(node_request.id)
                node_runs.append(
                    RunNodeResponse(
                        node_id=node_request.id,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        outputs=[],
                    )
                )
                continue

            port_values = _port_values(node, result)
            outputs[node_request.id] = port_values
            node_runs.append(
                RunNodeResponse(
                    node_id=node_request.id,
                    status="succeeded",
                    error=None,
                    outputs=[
                        await self._port_output_response(name, run_value)
                        for name, run_value in port_values.items()
                    ],
                )
            )

        status: Literal["succeeded", "failed"] = "failed" if failed else "succeeded"
        return RunResponse(status=status, node_runs=node_runs)

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
                port_values.append(run_value.value)

            if spec.variadic:
                values[name] = port_values
            elif len(port_values) == 1:
                values[name] = port_values[0]
            elif len(port_values) > 1:
                raise WorkbenchGraphError(
                    f"Node {node_request.id!r} input {name!r} accepts one "
                    f"connection, got {len(port_values)}"
                )
        return values

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


def _topological_order(
    nodes: list[RunNodeRequest],
    edges: list[RunEdgeRequest],
) -> list[RunNodeRequest]:
    by_id = {node.id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise WorkbenchGraphError("Duplicate node ids in graph")
    for edge in edges:
        if edge.from_node not in by_id or edge.to_node not in by_id:
            raise WorkbenchGraphError(
                f"Edge references unknown node {edge.from_node!r} -> {edge.to_node!r}"
            )

    incoming_count = {node.id: 0 for node in nodes}
    for edge in edges:
        incoming_count[edge.to_node] += 1

    queue = deque(node for node in nodes if incoming_count[node.id] == 0)
    ordered: list[RunNodeRequest] = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for edge in edges:
            if edge.from_node != node.id:
                continue
            incoming_count[edge.to_node] -= 1
            if incoming_count[edge.to_node] == 0:
                queue.append(by_id[edge.to_node])

    if len(ordered) != len(nodes):
        raise WorkbenchGraphError("Graph contains a cycle")
    return ordered


def _validate_edges(
    nodes_by_id: dict[str, Node[Any, Any, Any]],
    invocations_by_id: dict[str, NodeInvocation],
    edges: list[RunEdgeRequest],
    projectable_artifact_types: dict[ArtifactTypeKey, ArtifactTypeSpec],
) -> None:
    incoming_counts: dict[tuple[str, str], int] = {}
    for edge in edges:
        source_node = nodes_by_id[edge.from_node]
        target_node = nodes_by_id[edge.to_node]
        source_port = source_node.output_contract.ports.get(edge.from_port)
        if source_port is None:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} references unknown output "
                f"port {edge.from_port!r} on node {edge.from_node!r}"
            )
        target_port = target_node.input_contract.ports.get(edge.to_port)
        if target_port is None:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} references unknown input "
                f"port {edge.to_port!r} on node {edge.to_node!r}"
            )
        source_shape = effective_output_shape(
            source_node,
            invocations_by_id[edge.from_node],
            edge.from_port,
        )
        target_shape = effective_input_shape(
            target_node,
            invocations_by_id[edge.to_node],
            edge.to_port,
        )
        if edge.collection_mode == "map" and source_shape is not PortShape.MANY:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} uses collection mode 'map', "
                f"which requires a source with shape {PortShape.MANY.value!r}; "
                f"source is {source_shape.value!r}"
            )
        if source_shape != target_shape:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} has incompatible shapes: "
                f"source is {source_shape.value!r}, target expects "
                f"{target_shape.value!r}"
            )

        incoming_key = (edge.to_node, edge.to_port)
        incoming_counts[incoming_key] = incoming_counts.get(incoming_key, 0) + 1
        if not target_port.variadic and incoming_counts[incoming_key] > 1:
            raise WorkbenchGraphError(
                f"Node {edge.to_node!r} input {edge.to_port!r} accepts one "
                f"connection, got {incoming_counts[incoming_key]}"
            )

        if edge.projection is None:
            if source_port.produces != target_port.accepts:
                raise WorkbenchGraphError(
                    f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                    f"{edge.to_node!r}.{edge.to_port!r} cannot connect "
                    f"{source_port.produces.id}@"
                    f"{source_port.produces.schema_version} to "
                    f"{target_port.accepts.id}@{target_port.accepts.schema_version} "
                    f"without a declared field projection"
                )
            continue

        requested_path = tuple(edge.projection.path)
        projection = _field_projection_for(
            projectable_artifact_types,
            source_port.produces,
            requested_path,
        )
        if projection is None:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} requests undeclared "
                f"projection {'.'.join(requested_path)!r} on "
                f"{source_port.produces.id}@{source_port.produces.schema_version}"
            )
        if projection.target != target_port.accepts:
            raise WorkbenchGraphError(
                f"Edge {edge.from_node!r}.{edge.from_port!r} -> "
                f"{edge.to_node!r}.{edge.to_port!r} projects "
                f"{'.'.join(requested_path)!r} as {projection.target.id}@"
                f"{projection.target.schema_version}, but target expects "
                f"{target_port.accepts.id}@{target_port.accepts.schema_version}"
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
    if isinstance(run_value.value, ArtifactRef):
        return run_value.value.key()
    return ArtifactTypeKey(
        run_value.value.artifact_type,
        run_value.value.schema_version,
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
