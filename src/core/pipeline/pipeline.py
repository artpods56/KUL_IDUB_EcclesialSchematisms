"""Pipeline orchestration system for configurable ML model processing.

This module provides the core pipeline architecture for the AI Osrodek project, enabling
orchestrated processing of historical schematism documents through configurable machine
learning models and processing steps.

Architecture:
    The pipeline system is built around three key concepts:
    
    1. Pipeline: Main orchestrator that manages model configurations and coordinates
       the execution of processing phases in a dependency-aware manner.

    2. PipelinePhase: Containers that group related processing steps and handle
       their sequential execution with progress monitoring and type validation.

    3. Processing Steps: Individual operations that transform data (defined in
       the steps.base module).

Phase Types:
    - IngestionPhase: Entry points that load data from external sources
    - SampleProcessingPhase: Parallel processing of individual data items
    - DatasetProcessingPhase: Collection-level operations on entire datasets

Key Features:
    - Type Safety**: Automatic type validation between phases and steps
    - Dependency Management: Topological sorting of phases based on dependencies
    - Progress Monitoring: Built-in progress tracking with tqdm
    - Model Management: Centralized configuration and lifecycle management

Example:
    Basic pipeline setup and execution:
    
    ```python
    from core.models.ocr.model import OCRModel
    from core.models.llm.model import LLMModel

    from core.schemas.pipeline.data import PipelineData
    from core.pipeline.pipeline import Pipeline
    from core.pipeline.steps.ocr import OCRStep
    from core.pipeline.steps.extraction import TextExtractionStep

    # Configure models
    model_configs = {
        OCRModel: ocr_config,
        LLMModel: llm_config,
    }
    
    # Create pipeline
    pipeline = Pipeline(model_configs)
    
    # Add phases
    ingestion = IngestionPhase(steps=[load_images_step], name="load_data")
    processing = SampleProcessingPhase(
        steps=[ocr_step, extraction_step], 
        name="extract_text",
        depends_on=ingestion
    )
    
    pipeline.add_phases([ingestion, processing])
    
    # Execute
    results = pipeline.run()
    ```

Note:
    All pipeline components use structured logging for monitoring and debugging.
    Type mismatches between phases are detected early and raise descriptive errors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Iterable, Iterator, cast, Type, Any, Dict

from dataclasses import dataclass
from tqdm import tqdm
from structlog import get_logger

import core.schemas.configs  # type: ignore

from core.models.base import ConfigurableModel, ModelConfigMap
from core.pipeline.steps.base import (
    ProcessingStep,
    SampleProcessingStep,
    DatasetProcessingStep,
    IngestionProcessingStep,
)
from itertools import chain

from core.schemas.data.pipeline import PipelineData

logger = get_logger(__name__)


@dataclass
class PipelinePhase[TStep: ProcessingStep[Any, Any]](ABC):
    """Abstract base class for pipeline execution phases.

    Pipeline phases are containers that group related processing steps and manage
    their sequential execution. Each phase has a specific operational scope
    (ingestion, sample processing, or dataset processing) and handles progress
    monitoring, type validation, and dependency management.

    Args:
        TStep: Generic type parameter constraining the types of processing steps
            that can be contained within this phase.

    Attributes:
        steps (list[TStep]): Ordered list of processing steps to execute.
        name (str): Unique identifier for this phase.
        description (str | None): Optional human-readable description of the phase.
        depends_on (PipelinePhase[Any] | None): Optional dependency on another phase.
        input_type (type): Expected input data type, inferred from first step.
        output_type (type): Expected output data type, inferred from last step.

    Note:
        This is an abstract base class. Use the concrete subclasses:
        IngestionPhase, SampleProcessingPhase, or DatasetProcessingPhase.
    """

    steps: list[TStep]
    name: str
    description: str | None = None
    depends_on: PipelinePhase[Any] | None = None

    def __hash__(self) -> int:
        """Compute hash value based on the phase name.

        Returns:
            int: Hash value for this phase, enabling use in sets and as dict keys.
        """
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compare phases for equality based on hash values.

        Args:
            other: Object to compare against.

        Returns:
            bool: True if both objects have the same hash, False otherwise.
        """
        return hash(self) == hash(other)

    def _validate_steps(self) -> None:
        """Validate that all steps are compatible with this phase type.

        Ensures the phase contains at least one step and that all steps match
        the expected processing step type for this phase (ingestion, sample, or dataset).

        Raises:
            ValueError: If the phase contains no steps or contains steps of the wrong type.
            AssertionError: If step types don't match the phase type during runtime checks.
        """

        if len(self.steps) == 0:
            raise ValueError("Pipeline phase must contain at least one step.")

        for step in self.steps:

            if isinstance(self, IngestionPhase):
                assert isinstance(
                    step, IngestionProcessingStep
                ), "Should be ingestion step"
            elif isinstance(self, SampleProcessingPhase):
                assert isinstance(step, SampleProcessingStep), "Should be sample step"
            elif isinstance(self, DatasetProcessingPhase):
                assert isinstance(step, DatasetProcessingStep), "Should be dataset step"
            else:
                raise ValueError("Invalid pipeline phase")

    def _get_io_types(self) -> tuple[Any, Any]:
        """Determine input and output types from the step chain.

        Analyzes the first and last steps in the phase to determine the overall
        input and output types for the phase. This enables type validation
        between phases during pipeline construction.

        Returns:
            tuple[Any, Any]: Tuple of (input_type, output_type) where:
                - input_type: Expected input type from the first step
                - output_type: Produced output type from the last step
        """
        return self.steps[0].input_type, self.steps[-1].output_type

    def __post_init__(self) -> None:
        """Perform post-initialization validation and setup.

        Validates that required attributes are provided and infers the input/output
        types from the step configuration. This method is automatically called
        after dataclass initialization.

        Raises:
            ValueError: If the phase name is not provided.
        """
        if not self.name:
            raise ValueError("You must specify a name for the pipeline phase.")

        # self._validate_steps()

        self.input_type, self.output_type = self._get_io_types()

    @abstractmethod
    def execute(self, data: Iterable[Any]) -> Any:
        """Execute all steps in this phase sequentially.

        This method must be implemented by concrete subclasses to define how
        the steps within the phase are executed. The implementation should
        handle progress monitoring and step coordination.

        Args:
            data: Input data to be processed by this phase.

        Returns:
            Any: Processed data output from the last step in the phase.
        """
        ...


class IngestionPhase(PipelinePhase[IngestionProcessingStep[PipelineData]]):
    """Pipeline phase for data ingestion from external sources.

    Ingestion phases serve as entry points to the pipeline, loading data from
    external sources and converting it into the standard PipelineData file_format.
    Multiple ingestion steps can be combined to load from different sources
    simultaneously.

    The phase chains together all ingestion steps and flattens their output
    into a single stream of PipelineData items.

    Example:
        ```python
        ingestion = IngestionPhase(
            steps=[
                LoadImagesStep(directory="/path/to/images"),
                LoadTextFilesStep(directory="/path/to/texts")
            ],
            name="load_data",
            description="Load images and text files"
        )

        # Execute phase (data parameter is ignored for ingestion)
        pipeline_data = ingestion.execute([])
        ```
    """

    def execute(self, data: Iterable[Any]) -> Iterator[PipelineData]:
        """Execute all ingestion steps and chain their outputs.

        Args:
            data: Input data (ignored for ingestion phases as they generate
                their own data from external sources).

        Returns:
            Iterator[PipelineData]: Flattened stream of PipelineData items
                from all ingestion steps in this phase.
        """
        for step in tqdm(self.steps, desc=self.name, unit="step", position=0):
            assert isinstance(step, IngestionProcessingStep), "Should be ingestion step"

        return chain.from_iterable(step.iter_source() for step in self.steps)


class SampleProcessingPhase(
    PipelinePhase[SampleProcessingStep[PipelineData, PipelineData]]
):
    """Pipeline phase for sample-level data processing.

    Sample processing phases handle individual PipelineData items independently,
    making them ideal for operations that can be parallelized. Each step in the
    phase processes the entire batch before passing it to the next step.

    The phase chains steps sequentially, with each step's output becoming the
    input for the next step. Progress is monitored with a progress bar.

    Example:
        ```python
        processing = SampleProcessingPhase(
            steps=[
                OCRProcessingStep(model=ocr_model),
                LanguageDetectionStep(),
                LLMExtractionStep(model=llm_model)
            ],
            name="extract_data",
            description="OCR, detect language, and extract information",
            depends_on=ingestion_phase
        )

        # Execute phase
        processed_data = processing.execute(pipeline_data)
        ```
    """

    def execute(self, data: Iterable[PipelineData]) -> Iterable[PipelineData]:
        """Execute all sample processing steps sequentially.

        Processes the input data through each step in order, with each step
        transforming the entire batch before passing it to the next step.

        Args:
            data: Iterable of PipelineData items to process.

        Returns:
            Iterable[PipelineData]: Processed PipelineData items after all
                steps have been applied.
        """
        current = data
        for step in tqdm(self.steps, desc=self.name, unit="step", position=0):
            assert isinstance(
                step, SampleProcessingStep
            ), "Should be sample processing step"
            current = step.process_batch(current)
        return current


class DatasetProcessingPhase(PipelinePhase[DatasetProcessingStep[Any, Any]]):
    """Pipeline phase for dataset-level processing operations.

    Dataset processing phases operate on entire datasets at once, enabling
    operations that require global knowledge, aggregation, or coordination
    across all data items. These phases can transform data types (e.g.,
    from List[PipelineData] to pandas DataFrame) and perform exports.

    The phase chains steps sequentially, with each step having access to
    the complete dataset for global operations.

    Example:
        ```python
        export_phase = DatasetProcessingPhase(
            steps=[
                ConvertToDataFrameStep(),
                ApplySchemaMapping(),
                ExportToCSVStep(output_path="results.csv")
            ],
            name="export_results",
            description="Convert to DataFrame and export to CSV",
            depends_on=processing_phase
        )

        # Execute phase
        final_output = export_phase.execute(processed_data)
        ```
    """

    def execute(self, data: Any) -> Any:
        """Execute all dataset processing steps sequentially.

        Processes the complete dataset through each step in order, with each
        step having access to the entire dataset for global operations.

        Args:
            data: Complete dataset to process. Type depends on the specific
                steps in the phase (could be List[PipelineData], DataFrame, etc.).

        Returns:
            Any: Processed dataset after all steps have been applied. The output
                type depends on the final step in the phase.
        """
        current = data
        for step in tqdm(self.steps, desc=self.name, unit="step", position=0):
            assert isinstance(
                step, DatasetProcessingStep
            ), "Should be dataset processing step"
            current = step.process_dataset(current)

        return current


PipelinePhaseUnion = IngestionPhase | SampleProcessingPhase | DatasetProcessingPhase


class Pipeline[TConfigurableModel: ConfigurableModel, TPhase: PipelinePhaseUnion]:
    """Main orchestrator for ML model processing pipelines.

    The Pipeline class manages the complete lifecycle of document processing workflows,
    from data ingestion through ML model execution to final output generation. It
    provides centralized model configuration, dependency-aware phase execution,
    and comprehensive error handling.

    Key Features:
        - **Model Management**: Centralized configuration and initialization of ML models
        - **Phase Orchestration**: Dependency-aware execution of processing phases
        - **Type Safety**: Automatic validation of data flow between phases
        - **Progress Monitoring**: Built-in progress tracking and logging
        - **Error Handling**: Comprehensive error messages for debugging

    Args:
        TConfigurableModel: Generic type parameter for the configurable model types
            that this pipeline can manage.
        TPhase: Generic type parameter for the pipeline phase types that can be
            added to this pipeline.

    Attributes:
        model_configs (ModelConfigMap): Configuration mapping for ML models.
        batched (bool): Whether processing should be done in batches.
        batch_size (int): Size of batches when batched processing is enabled.

    Architecture:
        The pipeline executes in three main stages:

        1. **Model Setup**: Initialize configured models from their configurations
        2. **Phase Resolution**: Determine execution order based on dependencies
        3. **Sequential Execution**: Execute phases in dependency order with type validation

    Example:
        Complete pipeline setup and execution:

        ```python
        from core.models.ocr import OCRModel
        from core.models.llm import LLMModel

        # Configure models
        model_configs = {
            OCRModel: {
                'model_path': '/path/to/ocr/model',
                'confidence_threshold': 0.8
            },
            LLMModel: {
                'model_name': 'gpt-4',
                'temperature': 0.1
            }
        }

        # Create pipeline
        pipeline = Pipeline(model_configs, batched=True, batch_size=16)

        # Create processing phases
        ingestion = IngestionPhase(
            steps=[LoadImagesStep('/data/images')],
            name='load_data'
        )

        processing = SampleProcessingPhase(
            steps=[
                OCRStep(pipeline.get_model(OCRModel)),
                TextExtractionStep(pipeline.get_model(LLMModel))
            ],
            name='process_documents',
            depends_on=ingestion
        )

        export = DatasetProcessingPhase(
            steps=[ExportCSVStep('output.csv')],
            name='export_results',
            depends_on=processing
        )

        # Add phases and run
        pipeline.add_phases([ingestion, processing, export])
        results = pipeline.run()
        ```

    Dependency Management:
        Phases can depend on other phases using the `depends_on` attribute.
        The pipeline automatically resolves the execution order using topological
        sorting, ensuring dependencies are satisfied:

        ```python
        # Linear dependency chain
        phase_a = IngestionPhase(steps=[...], name='a')
        phase_b = SampleProcessingPhase(steps=[...], name='b', depends_on=phase_a)
        phase_c = DatasetProcessingPhase(steps=[...], name='c', depends_on=phase_b)

        # Pipeline automatically orders as: a -> b -> c
        pipeline.add_phases([phase_c, phase_a, phase_b])  # Order doesn't matter
        ```

    Error Handling:
        The pipeline provides detailed error messages for common issues:
        - Model configuration errors with specific model names
        - Type mismatches between phases with expected/actual types
        - Dependency cycles and disconnected phase graphs
        - Missing models when requested by processing steps

    Note:
        All pipeline operations use structured logging for monitoring and debugging.
        Phase execution includes automatic progress tracking with descriptive names.
    """

    def __init__(
        self, model_configs: ModelConfigMap, batched: bool = False, batch_size: int = 10
    ):
        """Initialize a new pipeline with model configurations.

        Sets up the pipeline with the provided model configurations and initializes
        all configured models. The pipeline can optionally support batched processing
        for improved performance with large datasets.

        Args:
            model_configs: Mapping from model classes to their configuration objects.
                Each model will be initialized using its from_config class method.
            batched: Whether to enable batched processing for improved performance.
                Currently not fully implemented in all steps.
            batch_size: Number of items to process in each batch when batched
                processing is enabled.

        Raises:
            RuntimeError: If any model fails to initialize from its configuration.
        """
        self.model_configs = model_configs

        self._models: Mapping[Type[ConfigurableModel], ConfigurableModel] = {}
        if self.model_configs:
            self._models = self._setup_models()

        self._phases: Dict[str, TPhase] = {}

        self.batched = batched
        self.batch_size = batch_size

    def _setup_models(self) -> Mapping[Type[ConfigurableModel], ConfigurableModel]:
        """Initialize all configured models from their configurations.

        Iterates through the model configurations and instantiates each model
        using its from_config class method. Provides detailed error messages
        if any model fails to initialize.

        Returns:
            Mapping[Type[ConfigurableModel], ConfigurableModel]: Dictionary mapping
                model classes to their initialized instances.

        Raises:
            RuntimeError: If no model configs were provided or if any model fails
                to initialize. The original exception is chained for debugging.
        """
        if not self.model_configs:
            raise RuntimeError("No model configs were provided for the pipeline.")

        models: dict[Type[ConfigurableModel], ConfigurableModel] = {}
        for model_cls, config in self.model_configs.items():
            try:
                models[model_cls] = model_cls.from_config(config)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize model {model_cls.__name__}"
                ) from e
        return models

    def _validate_phases(self, phases: list[TPhase]) -> None:
        """Validate type compatibility between consecutive pipeline phases.

        Ensures that the output type of each phase matches the input type of the
        next phase in the sequence, preventing runtime type errors during execution.

        Args:
            phases: Ordered list of pipeline phases to validate. Each phase's
                output type must be compatible with the next phase's input type.

        [TODO]: Implement this
        """
        pass

    def get_model[TModel: ConfigurableModel](self, model_cls: type[TModel]) -> TModel:
        """Retrieve a configured model instance by its class.

        This method is typically used by processing steps to access the models
        they need for their operations. The model must have been configured when
        the pipeline was initialized.

        Args:
            model_cls: The class of the model to retrieve. Must be a subclass
                of ConfigurableModel that was included in the model_configs
                passed to the pipeline constructor.

        Returns:
            TModel: The initialized model instance of the requested type.

        Raises:
            KeyError: If the requested model class was not configured in this
                pipeline. The error message includes the model class name.
        """
        if model_cls not in self._models:
            raise KeyError(
                f"Model {model_cls.__name__} is not configured in this pipeline."
            )
        return cast(TModel, self._models[model_cls])

    def add_phases(self, phases: list[TPhase]) -> None:
        """Add multiple phases to this pipeline.

        Phases can be added in any order as the pipeline will automatically
        resolve the execution order based on their dependencies when run() is called.

        Args:
            phases: List of phases to add to the pipeline. Each phase must have
                a unique name and can optionally depend on other phases.

        Raises:
            ValueError: If a phase with the same name already exists in the pipeline.
        """
        for phase in phases:
            # phase._validate_steps()
            if phase in self._phases:
                raise ValueError(f"Phase {phase.name} already exists in the pipeline.")

            self._phases[phase.name] = phase

    def _resolve_execution_order(self) -> list[TPhase]:
        """Determine the linear execution order using topological sorting.

        Analyzes the dependency graph formed by phase dependencies and determines
        a valid execution order. The pipeline supports only linear dependency chains
        (no branching or merging).

        Returns:
            list[TPhase]: Ordered list of phases ready for execution, starting
                with the ingestion phase.

        Raises:
            RuntimeError: In several cases:
                - No phases were added to the pipeline
                - Multiple terminal phases exist (should be exactly one)
                - Circular dependencies are detected
                - Phases don't form a single connected chain
                - First phase is not an ingestion phase
        """
        if not self._phases:
            raise RuntimeError("No phases were added to the pipeline.")

        all_phases = set(self._phases.values())
        dependencies = {p.depends_on for p in all_phases if p.depends_on is not None}
        tail_phases = [p for p in all_phases if p not in dependencies]

        if len(tail_phases) != 1:
            raise RuntimeError(
                f"Expected 1 terminal phase, but found {len(tail_phases)}. "
                "The pipeline should be a single linear phases_chain."
            )
        phases_chain: list[TPhase] = []
        current_phase = tail_phases[0]
        while current_phase:
            if current_phase in phases_chain:
                raise RuntimeError("A cycle was detected in the dependency graph.")

            phases_chain.append(current_phase)

            current_phase = cast(TPhase, current_phase.depends_on)

        phases_chain.reverse()

        if len(phases_chain) != len(self._phases):
            raise RuntimeError(
                "The pipeline phases do not form a single connected phases_chain."
            )

        assert isinstance(
            phases_chain[0], IngestionPhase
        ), "First phase must be ingestion phase."

        return phases_chain

    def run(self) -> Any:
        """Execute the complete pipeline from ingestion to final output.

        Runs all phases in dependency order, starting with data ingestion and
        proceeding through sample processing and dataset operations. Each phase
        receives the output of the previous phase as its input.

        Returns:
            Any: Final processed output from the last phase in the pipeline.
                The type depends on the final phase type:
                - IngestionPhase: Iterator[PipelineData]
                - SampleProcessingPhase: Iterable[PipelineData]
                - DatasetProcessingPhase: Any (depends on the specific steps)

        Raises:
            RuntimeError: If no phases are configured or if phases don't form
                a valid dependency chain.
            DataTypeMismatchError: If adjacent phases have incompatible types.
            AssertionError: If the first phase is not an ingestion phase.
        """
        phases = self._resolve_execution_order()

        self._validate_phases(phases)

        current_data: Any = []
        for i, phase in enumerate(phases):
            current_data = phase.execute(current_data)

        return current_data
