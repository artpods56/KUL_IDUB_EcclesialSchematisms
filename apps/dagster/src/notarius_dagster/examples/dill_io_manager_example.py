"""Example usage of DillIOManager in Dagster pipelines.

This example demonstrates how to use the DillIOManager for storing
complex objects that standard pickle cannot handle.
"""

from dagster import Definitions, asset, materialize

from notarius_dagster.dill_io_manager import dill_io_manager
from notarius.shared.constants import TMP_DIR


# Example 1: Simple asset with lambda functions
@asset
def data_with_lambda() -> dict:
    """Asset that produces data containing lambda functions.

    Standard pickle would fail on this, but dill handles it perfectly.
    """
    return {
        "preprocessor": lambda x: x.strip().lower(),
        "validator": lambda x: len(x) > 0,
        "transformer": lambda x: x.replace(" ", "_"),
    }


@asset
def processed_data(data_with_lambda: dict) -> list[str]:
    """Asset that uses the lambda functions from the previous asset."""
    test_strings = ["  Hello World  ", "  Python  ", "  Dagster  "]

    preprocessor = data_with_lambda["preprocessor"]
    validator = data_with_lambda["validator"]
    transformer = data_with_lambda["transformer"]

    results = []
    for s in test_strings:
        processed = preprocessor(s)
        if validator(processed):
            results.append(transformer(processed))

    return results


# Example 2: Asset with closure
@asset
def create_counter() -> callable:
    """Asset that produces a closure (function with captured state).

    Closures are difficult for pickle but easy for dill.
    """
    count = 0

    def counter():
        nonlocal count
        count += 1
        return count

    return counter


@asset
def use_counter(create_counter: callable) -> list[int]:
    """Asset that uses the counter closure."""
    counter = create_counter
    return [counter() for _ in range(5)]


# Example 3: Asset with custom class containing functions
class DataProcessor:
    """Example class with embedded functions."""

    def __init__(self, name: str):
        self.name = name
        self.operations = {
            "double": lambda x: x * 2,
            "square": lambda x: x**2,
            "increment": lambda x: x + 1,
        }

    def process(self, value: int, operation: str) -> int:
        """Apply an operation to a value."""
        return self.operations[operation](value)


@asset
def processor_factory() -> DataProcessor:
    """Asset that creates a processor instance."""
    return DataProcessor(name="example_processor")


@asset
def processed_results(processor_factory: DataProcessor) -> dict:
    """Asset that uses the processor."""
    processor = processor_factory
    return {
        "doubled": processor.process(5, "double"),
        "squared": processor.process(5, "square"),
        "incremented": processor.process(5, "increment"),
    }


# Define the Dagster project with DillIOManager
defs = Definitions(
    assets=[
        # Example 1
        data_with_lambda,
        processed_data,
        # Example 2
        create_counter,
        use_counter,
        # Example 3
        processor_factory,
        processed_results,
    ],
    resources={
        "io_manager": dill_io_manager(base_dir=str(TMP_DIR / "dagster_dill_examples")),
    },
)


# Example usage script
if __name__ == "__main__":
    print("=" * 60)
    print("DillIOManager Example - Lambdas and Functions")
    print("=" * 60)

    # Example 1: Lambda functions
    print("\nExample 1: Processing data with lambdas")
    result = materialize(
        [data_with_lambda, processed_data],
        resources={"io_manager": dill_io_manager()},
    )

    if result.success:
        output = result.output_for_node("processed_data")
        print(f"Processed strings: {output}")

    # Example 2: Closures
    print("\nExample 2: Using closures")
    result = materialize(
        [create_counter, use_counter],
        resources={"io_manager": dill_io_manager()},
    )

    if result.success:
        output = result.output_for_node("use_counter")
        print(f"Counter values: {output}")

    # Example 3: Custom class with embedded functions
    print("\nExample 3: Custom processor class")
    result = materialize(
        [processor_factory, processed_results],
        resources={"io_manager": dill_io_manager()},
    )

    if result.success:
        output = result.output_for_node("processed_results")
        print(f"Processing results: {output}")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
