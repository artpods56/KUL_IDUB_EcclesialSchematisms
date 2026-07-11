from dataclasses import dataclass
from pydantic import BaseModel

from notarius_core.domain.models.completions import BaseProviderResponse


@dataclass(frozen=True)
class OpenAIResponse[T: BaseModel](BaseProviderResponse[T]):
    structured_response: T | None
    text_response: str | None
