"""Base factory class for test data generation.

This module provides the foundation for all test factories, implementing
a generic factory pattern that creates real objects with sensible defaults.
"""

from typing import TypeVar, Generic, Any


T = TypeVar('T')


class BaseFactory(Generic[T]):
    """Base factory class for creating test objects.

    This factory provides a generic interface for creating test data objects
    with sensible defaults, eliminating the need for mock-heavy testing.

    All concrete factories should extend this class and implement the build() method.

    Example:
        class SchematismEntryFactory(BaseFactory[SchematismEntry]):
            _counter = 0

            @classmethod
            def build(cls, parish: str | None = None, **kwargs) -> SchematismEntry:
                cls._counter += 1
                return SchematismEntry(
                    parish=parish or f"Test Parish {cls._counter}",
                    **kwargs
                )
    """

    _counter: int = 0

    @classmethod
    def build(cls, **kwargs) -> T:
        """Build a single instance of the target type.

        Subclasses must implement this method to define how objects are created.

        Args:
            **kwargs: Attributes to override default values

        Returns:
            A new instance of type T

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement the build() method"
        )

    @classmethod
    def build_batch(cls, size: int, **kwargs) -> list[T]:
        """Build multiple instances with the same parameters.

        Args:
            size: Number of instances to create
            **kwargs: Attributes to apply to all instances

        Returns:
            A list of instances of type T

        Example:
            items = SchematismEntryFactory.build_batch(5, deanery="Test Deanery")
            # Creates 5 SchematismEntry objects, all with deanery="Test Deanery"
        """
        return [cls.build(**kwargs) for _ in range(size)]

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the internal counter for this factory.

        Useful for ensuring consistent IDs across tests or test runs.
        """
        cls._counter = 0

    @classmethod
    def create(cls, **kwargs) -> T:
        """Create and persist an instance (for integration tests).

        Default implementation just calls build(). Subclasses can override
        to add persistence logic (e.g., saving to database).

        Args:
            **kwargs: Attributes to override default values

        Returns:
            A new (and potentially persisted) instance of type T
        """
        return cls.build(**kwargs)
