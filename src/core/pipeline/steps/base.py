"""Base classes for pipeline processing steps.

Provides abstract base classes for implementing processing steps in the AI Osrodek
pipeline system for historical schematism document processing.

Classes:
    IngestionProcessingStep: Entry points that load data from external sources.
    SampleProcessingStep: Item-level processors for individual data items.
    DatasetProcessingStep: Collection-level processors for entire datasets.
    
All steps use generic type parameters for type-safe pipeline composition.
Each step receives a structured logger configured with the class name.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Iterator, get_type_hints, Iterable, List, Any, get_origin, get_args

from structlog import BoundLogger, get_logger
from structlog.typing import BindableLogger
from tqdm import tqdm


class ProcessingStep[InT, OutT](ABC):
    """Abstract base class for all pipeline processing steps.
    
    Provides foundation for implementing processing steps with automatic type inference
    and structured logging.
    
    Args:
        InT: Input data type.
        OutT: Output data type.
    
    Attributes:
        input_type: Expected input data type, inferred from method annotations.
        output_type: Expected output data type, inferred from method annotations.
        logger: Structured logger instance for this step.
    """

    def __init__(self):
        """Initialize the processing step with automatic type inference."""
        super().__init__()
        self.input_type, self.output_type = None, None #self._infer_io_types() TODO: implement this

    def __init_subclass__(cls) -> None:
        """Initialize subclass with dedicated structured logger."""
        super().__init_subclass__()
        cls._logger = get_logger(cls.__name__)

    @abstractmethod
    def _get_main_method(self) -> Callable[..., Any]:
        """Return the main processing method for type inference.
        
        Returns:
            The main processing method containing type annotations for inference.
        """
        ...

    def _extract_concrete_type(self, type_hint: Any) -> type:
        """Extract concrete type from type hints, handling generics.
        
        Args:
            type_hint: The type hint to extract from (may be generic like Iterator[dict])
            
        Returns:
            Concrete type suitable for beartype validation
        """
        # Handle None type
        if type_hint is type(None) or type_hint is None:
            return type(None)
            
        # Handle generic types like Iterator[dict], List[str], etc.
        origin = get_origin(type_hint)
        if origin is not None:
            args = get_args(type_hint)
            if args:
                # For generic types like Iterator[dict], return the inner type (dict)
                return args[0] if isinstance(args[0], type) else object
            else:
                # For generic types without args, return object
                return object
                
        # Handle concrete types
        if isinstance(type_hint, type):
            return type_hint
            
        # Fallback for any other cases
        return object

    def _infer_io_types(self) -> tuple[type, type]:
        """Infer input and output types from method annotations.
        
        Returns:
            Tuple of (input_type, output_type). For SampleProcessingStep,
            types are wrapped in List[T] for batch processing.
        """

        hints = get_type_hints(self._get_main_method())

        if isinstance(self, SampleProcessingStep):
            for param_name, hint in hints.items():
                hints[param_name] = List[hint]

        param_names = [k for k in hints.keys() if k != "return"]
        input_type_hint = hints[param_names[0]] if param_names else type(None)
        output_type_hint = hints.get("return", object)
        
        # Extract concrete types for beartype compatibility
        input_type = self._extract_concrete_type(input_type_hint)
        output_type = self._extract_concrete_type(output_type_hint)
        
        return input_type, output_type


    @property
    def logger(self) -> BoundLogger:
        """Get the structured logger for this processing step.
        
        Returns:
            Structured logger instance configured with this step's class name.
        """
        return self.__class__._logger


class DatasetProcessingStep[InT, OutT](ProcessingStep[InT, OutT]):
    """Abstract base class for dataset-level processing steps.
    
    Operates on entire collections of data items at once, ideal for operations
    requiring global knowledge, aggregation, or coordination across items.
    
    Args:
        InT: Input dataset type.
        OutT: Output dataset type.
    """

    def _get_main_method(self) -> Callable[[InT], OutT]:
        """Return the main processing method for type inference.
        
        Returns:
            The process_dataset method used for type inference.
        """
        return self.process_dataset

    @abstractmethod
    def process_dataset(self, dataset: InT) -> OutT:
        """Process an entire dataset.
        
        Args:
            dataset: The complete dataset to process.
        
        Returns:
            The processed dataset.
        """
        ...

class SampleProcessingStep[InT, OutT](ProcessingStep[InT, OutT]):
    """Abstract base class for sample-level processing steps.
    
    Operates on individual data items independently, enabling parallelization.
    Automatically provides batch processing by mapping process_sample over collections.
    
    Args:
        InT: Individual input item type.
        OutT: Individual output item type.
    """
    
    def _get_main_method(self) -> Callable[[InT], OutT]:
        """Return the main processing method for type inference.
        
        Returns:
            The process_sample method used for type inference.
        """
        return self.process_sample

    @abstractmethod
    def process_sample(self, data: InT) -> OutT:
        """Process a single data item.
        
        Args:
            data: A single input data item to process.
        
        Returns:
            The processed output item.
        """
        ...

    def process_batch(self, data: Iterable[InT]) -> Iterable[OutT]:
        """Process a batch of items by mapping process_sample over the collection.
        
        Args:
            data: An iterable collection of input items to process.
        
        Returns:
            A list containing the processed output items in the same order.
        """
        processed_dataset = []
        for i, item in enumerate(data):
            self.logger.info(f"Processing sample {i+1}")
            processed_dataset.append(self.process_sample(item))
        return processed_dataset


class IngestionProcessingStep[OutT](ProcessingStep[Any, OutT]):
    """Abstract base class for data ingestion processing steps.
    
    Entry points to the pipeline that load and convert raw data from external
    sources into standardized pipeline formats. Uses generators for memory efficiency.
    
    Args:
        OutT: Output data type that this ingestion step will produce.
    """

    def __init__(self):
        """Initialize the ingestion processing step."""
        super().__init__()

    def _get_main_method(self) -> Callable[[], Iterator[OutT]]:
        """Return the main processing method for type inference.
        
        Returns:
            The iter_source method used for type inference.
        """
        return self.iter_source

    @abstractmethod
    def iter_source(self) -> Iterator[OutT]:
        """Generate data items from the external source.
        
        Returns:
            An iterator yielding data items to be passed to subsequent pipeline steps.
        """
        ...

