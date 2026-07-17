from typing import cast

from notarius_core.runtime.persistence import InlineModelOutputWriter
from notarius_core.runtime.resolvers import InlineModelResolver, Resolver

from notarius_plugin_llm import mistral, openai_compatible
from notarius_plugin_llm.artifacts import (
    COMPLETION,
    STRUCTURED_RESPONSE,
    CompletionPayload,
    StructuredResponsePayload,
)
from notarius_plugin_llm.declaration import LLM


_NODE_MODULES = (mistral, openai_compatible)

LLM.register_artifact_type(STRUCTURED_RESPONSE)
LLM.register_artifact_type(COMPLETION)
LLM.register_resolver(
    lambda context: cast(
        Resolver[object],
        InlineModelResolver(
            source=STRUCTURED_RESPONSE.key,
            target=StructuredResponsePayload,
            uow=context.uow,
        ),
    )
)
LLM.register_writer(
    lambda context: InlineModelOutputWriter(
        artifact_type=STRUCTURED_RESPONSE.key,
        model=StructuredResponsePayload,
        uow=context.uow,
    )
)
LLM.register_resolver(
    lambda context: cast(
        Resolver[object],
        InlineModelResolver(
            source=COMPLETION.key,
            target=CompletionPayload,
            uow=context.uow,
        ),
    )
)
LLM.register_writer(
    lambda context: InlineModelOutputWriter(
        artifact_type=COMPLETION.key,
        model=CompletionPayload,
        uow=context.uow,
    )
)


__all__ = ["LLM"]
