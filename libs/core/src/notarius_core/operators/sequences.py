from typing import Annotated, final, override

from pydantic import Field

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    JsonObject,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import (
    ArtifactTypeVariable,
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
)
from notarius_core.plugins import Plugin


SEQUENCES = Plugin(
    slug="builtin.sequence",
    title="Sequence",
)


COLLECTED_ARTIFACT_TYPE = ArtifactTypeVariable("T")


class CollectInput(NodeInput):
    items: Annotated[
        list[ArtifactRef | ArtifactRefSequence],
        InPort(
            COLLECTED_ARTIFACT_TYPE,
            variadic=True,
            instance_plugs=True,
        ),
        Field(
            min_length=1,
            description="Artifacts and sequences in connection order.",
        ),
    ]


class CollectOutput(NodeOutput):
    items: Annotated[
        ArtifactRefSequence,
        OutPort(COLLECTED_ARTIFACT_TYPE),
        Field(description="One sequence containing every input artifact."),
    ]


@SEQUENCES.node(
    operator_id="sequence.collect",
    version=1,
    title="Collect",
)
@final
class CollectNode(Node[NoConfig, CollectInput, CollectOutput]):
    """Collects homogeneous artifact refs without rewriting their payloads."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: CollectInput,
        /,
    ) -> CollectOutput:
        first = inputs.items[0]
        if isinstance(first, ArtifactRef):
            artifact_type = first.key()
        else:
            artifact_type = ArtifactTypeKey(
                first.artifact_type,
                first.schema_version,
            )

        item_refs: list[ArtifactRef] = []
        collect_segments: list[JsonObject] = []
        ordered = True
        for input_index, source in enumerate(inputs.items):
            if isinstance(source, ArtifactRef):
                source_type = source.key()
            else:
                source_type = ArtifactTypeKey(
                    source.artifact_type,
                    source.schema_version,
                )
            if source_type != artifact_type:
                raise ValueError(
                    "Collect inputs must share one artifact type; expected "
                    f"{artifact_type.id}@{artifact_type.schema_version}, got "
                    f"{source_type.id}@{source_type.schema_version} at input "
                    f"{input_index}"
                )

            start_index = len(item_refs)
            if isinstance(source, ArtifactRef):
                item_refs.append(source)
                item_count = 1
                source_kind = "single"
            else:
                item_refs.extend(source.item_refs)
                item_count = len(source.item_refs)
                source_kind = "sequence"
                if not source.ordered:
                    ordered = False
            collect_segments.append(
                {
                    "input_index": input_index,
                    "start_index": start_index,
                    "item_count": item_count,
                    "source_kind": source_kind,
                }
            )

        return CollectOutput(
            items=ArtifactRefSequence(
                artifact_type=artifact_type.id,
                schema_version=artifact_type.schema_version,
                item_refs=item_refs,
                ordered=ordered,
                metadata={"collect_segments": collect_segments},
            )
        )
