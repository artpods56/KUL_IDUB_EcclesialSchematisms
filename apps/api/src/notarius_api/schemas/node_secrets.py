from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class NodeSecretApiModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ConfigureNodeSecretRequest(NodeSecretApiModel):
    value: SecretStr
    expected_graph_revision: int = Field(ge=1)


class NodeSecretStatusResponse(NodeSecretApiModel):
    node_id: str
    name: str
    configured: bool


class GraphNodeSecretsResponse(NodeSecretApiModel):
    graph_id: UUID
    graph_revision: int
    secrets: list[NodeSecretStatusResponse]
