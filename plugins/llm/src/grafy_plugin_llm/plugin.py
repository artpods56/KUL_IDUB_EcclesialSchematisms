from grafy_core.artifacts import Artifact
from grafy_core.runtime.persistence import InlineModelOutputWriter
from grafy_core.runtime.resolvers import InlineModelResolver

from grafy_plugin_llm import mistral, openai_compatible
from grafy_plugin_llm.artifacts import (
    COMPLETION,
    STRUCTURED_RESPONSE,
    CompletionPayload,
    StructuredResponsePayload,
)
from grafy_plugin_llm.declaration import LLM


_NODE_MODULES = (mistral, openai_compatible)

LLM.register(
    Artifact(
        spec=STRUCTURED_RESPONSE,
        resolver=lambda context: InlineModelResolver(
            source=STRUCTURED_RESPONSE.key,
            target=StructuredResponsePayload,
            uow=context.uow,
        ),
        writer=lambda context: InlineModelOutputWriter(
            artifact_type=STRUCTURED_RESPONSE.key,
            model=StructuredResponsePayload,
            uow=context.uow,
        ),
    )
)
LLM.register(
    Artifact(
        spec=COMPLETION,
        resolver=lambda context: InlineModelResolver(
            source=COMPLETION.key,
            target=CompletionPayload,
            uow=context.uow,
        ),
        writer=lambda context: InlineModelOutputWriter(
            artifact_type=COMPLETION.key,
            model=CompletionPayload,
            uow=context.uow,
        ),
    )
)


__all__ = ["LLM"]
