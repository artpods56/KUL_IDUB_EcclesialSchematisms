"""Run one saved LLM node with its materialized prompt through the real runtime."""

import argparse
import asyncio
from uuid import UUID

from pydantic import BaseModel

from grafy_core.artifacts import ArtifactRef, ArtifactRefSequence
from grafy_core.prompt_contracts import PROMPT_MESSAGE, PromptMessage
from grafy_api.app_state import get_resources
from grafy_api.main import create_app
from grafy_api.settings import Settings
from grafy_api.v1.models import (
    ArtifactTypeBindingModel,
    ArtifactTypeKeyResponse,
    PluginReleasePinModel,
)
from grafy_api.v1.routes.executions.models import (
    ArtifactConversionRequest,
    FieldProjectionRequest,
    PinnedOutputRequest,
    RunEdgeRequest,
    RunInputPlugRequest,
    RunNodeRequest,
    RunRequest,
)


class LlmImageVerificationResult(BaseModel):
    status: str
    node_id: str
    plugin_slug: str
    plugin_revision: int
    output_port: str
    output_artifact_type: str
    output_artifact_id: UUID
    prompt_artifact_id: UUID | None
    prompt_image_count: int


async def verify(
    *,
    workspace_id: UUID,
    graph_id: UUID,
    node_id: str,
    plugin_revision: int | None,
    prompt_artifact_id: UUID | None,
) -> LlmImageVerificationResult:
    settings = Settings(
        require_single_api_owner=False,
        log_level="WARNING",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        resources = get_resources(app)
        graph = await resources.saved_graphs.get(workspace_id, graph_id)
        target = next(
            (candidate for candidate in graph.document.nodes if candidate.id == node_id),
            None,
        )
        if target is None:
            raise RuntimeError(f"Graph {graph_id} does not contain node {node_id!r}")
        if target.plugin_release_pin is None:
            raise RuntimeError(f"Node {node_id!r} does not have a Plugin release pin")

        release_pin = PluginReleasePinModel.from_saved_pin(target.plugin_release_pin)
        if plugin_revision is not None:
            release_pin = release_pin.model_copy(
                update={"revision": plugin_revision},
            )

        incoming_edges = tuple(
            edge
            for edge in graph.document.edges
            if edge.enabled and edge.to_node == node_id
        )
        if not incoming_edges:
            raise RuntimeError(f"Node {node_id!r} has no connected inputs")
        materializations = await resources.materializations.list_for_graph(
            workspace_id,
            graph_id,
            graph.revision,
        )
        materializations_by_node = {
            materialization.node_id: materialization
            for materialization in materializations
        }
        prompt_override: ArtifactRefSequence | None = None
        prompt_image_count = 0
        if prompt_artifact_id is not None:
            prompt_artifact = await resources.artifacts.get(
                workspace_id,
                prompt_artifact_id,
            )
            if prompt_artifact is None:
                raise RuntimeError(
                    f"Prompt artifact {prompt_artifact_id} does not exist"
                )
            prompt = PromptMessage.model_validate(prompt_artifact.inline_payload)
            prompt_image_count = len(prompt.image_refs)
            if prompt_image_count == 0:
                raise RuntimeError(
                    f"Prompt artifact {prompt_artifact_id} does not reference images"
                )
            prompt_override = ArtifactRefSequence.from_key(
                key=PROMPT_MESSAGE.key,
                item_refs=[prompt_artifact.ref()],
            )
        pinned_outputs: list[PinnedOutputRequest] = []
        for edge in incoming_edges:
            source = materializations_by_node.get(edge.from_node)
            value = source.outputs.get(edge.from_port) if source is not None else None
            if edge.to_port == "messages" and prompt_override is not None:
                value = prompt_override
            if value is None:
                raise RuntimeError(
                    f"Input {edge.from_node!r}.{edge.from_port!r} for node "
                    f"{node_id!r} has no materialized output at graph revision "
                    f"{graph.revision}"
                )
            pinned_outputs.append(
                PinnedOutputRequest(
                    from_node=edge.from_node,
                    from_port=edge.from_port,
                    value=value,
                )
            )

        execution = await resources.run_graph.run(
            workspace_id,
            RunRequest(
                nodes=[
                    RunNodeRequest(
                        id=target.id,
                        operator_id=target.operator_id,
                        operator_version=target.operator_version,
                        config=target.config_dict(),
                        input_plugs=[
                            RunInputPlugRequest(id=plug.id, port=plug.port)
                            for plug in target.input_plugs
                        ],
                        artifact_type_bindings=[
                            ArtifactTypeBindingModel(
                                variable=binding.variable,
                                artifact_type=ArtifactTypeKeyResponse.from_key(
                                    binding.artifact_type
                                ),
                            )
                            for binding in target.artifact_type_bindings
                        ],
                        plugin_release=release_pin,
                    )
                ],
                edges=[
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
                    for edge in incoming_edges
                ],
                pinned_outputs=pinned_outputs,
                scope="selected",
                secret_graph_id=graph_id,
                secret_graph_revision=graph.revision,
            ),
        )
        result = next(
            (candidate for candidate in execution.node_results if candidate.node_id == node_id),
            None,
        )
        if result is None:
            raise RuntimeError(f"Execution did not return node {node_id!r}")
        if result.status != "succeeded":
            raise RuntimeError(result.error or f"Node {node_id!r} failed")
        completion = result.outputs.get("completion")
        if not isinstance(completion, ArtifactRef):
            raise RuntimeError(f"Node {node_id!r} did not produce one completion")
        return LlmImageVerificationResult(
            status=execution.status,
            node_id=node_id,
            plugin_slug=release_pin.slug,
            plugin_revision=release_pin.revision,
            output_port="completion",
            output_artifact_type=completion.artifact_type,
            output_artifact_id=completion.artifact_id,
            prompt_artifact_id=prompt_artifact_id,
            prompt_image_count=prompt_image_count,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=UUID)
    parser.add_argument("--graph", required=True, type=UUID)
    parser.add_argument("--node", required=True)
    parser.add_argument("--plugin-revision", type=int)
    parser.add_argument("--prompt-artifact", type=UUID)
    arguments = parser.parse_args()
    result = asyncio.run(
        verify(
            workspace_id=arguments.workspace,
            graph_id=arguments.graph,
            node_id=arguments.node,
            plugin_revision=arguments.plugin_revision,
            prompt_artifact_id=arguments.prompt_artifact,
        )
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
