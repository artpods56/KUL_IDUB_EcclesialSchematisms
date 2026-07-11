import abc
from collections.abc import Sequence
from typing import Protocol, final


class Validator[T](Protocol):
    @abc.abstractmethod
    async def validate(self, data: T) -> None:
        """Validate input, raising a domain exception on failure."""


@final
class ComposedValidator[T](Validator[T]):
    def __init__(self, validators: Sequence[Validator[T]]):
        self.validators = validators

    async def validate(self, data: T) -> None:
        for validator in self.validators:
            await validator.validate(data)

