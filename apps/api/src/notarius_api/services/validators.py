from dataclasses import dataclass
from uuid import UUID

from notarius_api.validator import Validator
from notarius_core.domain.errors import ValidationError
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort


@dataclass(frozen=True)
class ProjectScopedId:
    project_id: UUID
    related_id: UUID


class NameRequiredValidator(Validator[object]):
    async def validate(self, data: object) -> None:
        name = getattr(data, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Name is required")


class ProjectExistsValidator(Validator[UUID]):
    def __init__(self, uow: StudioUnitOfWorkPort):
        self.uow = uow

    async def validate(self, data: UUID) -> None:
        if await self.uow.projects.get(data) is None:
            raise ValidationError(f"Project does not exist: {data}")
