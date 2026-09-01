from grafy_core.artifacts import Artifact
from grafy_core.runtime.persistence import InlineModelOutputWriter
from grafy_core.runtime.resolvers import InlineModelResolver

from grafy_plugin_llm import openai_compatible, prompt
from grafy_plugin_llm.artifacts import COMPLETION, CompletionPayload
from grafy_plugin_llm.declaration import LLM


_NODE_MODULES = (prompt, openai_compatible)

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
