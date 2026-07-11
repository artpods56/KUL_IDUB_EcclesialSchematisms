from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast, override

import dagster as dg
import pandas as pd
import wandb
import weave
from dagster import ConfigurableResource
from pydantic import PrivateAttr

from notarius.application.ports.outbound.cached_engine import ResponseValidator
from notarius.domain.protocols import BaseResponse
from notarius.infrastructure.cache.backends.llm import create_llm_cache_backend
from notarius.infrastructure.config.constants import ConfigType, ModelsConfigSubtype
from notarius.infrastructure.config.manager import config_manager
from notarius.infrastructure.llm.engine_adapter import CachedLLMEngine, LLMEngine
from notarius.infrastructure.ml_models.lmv3.engine_adapter import LMv3Engine
from notarius.infrastructure.ocr.engine_adapter import OCREngine
from notarius.infrastructure.persistence.storage import ImageRepository
from notarius.shared.constants import PDF_SOURCE_DIR
from notarius.schemas.configs import (
    BaseLMv3ModelConfig,
    LLMEngineConfig,
    PytesseractOCRConfig,
)


class PdfFilesResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    pdf_dir: str = str(PDF_SOURCE_DIR)

    def get_pdf_paths(self) -> list[Path]:
        return list(Path(self.pdf_dir).glob("**/*.pdf"))


class ExcelWriterResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    writing_path: str

    @contextmanager
    def get_writer(self, file_name: str):
        writer_path = Path(self.writing_path) / Path(file_name)

        writer = pd.ExcelWriter(writer_path)

        try:
            yield writer
        finally:
            writer.close()


class WandBRunResource(
    ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    run_name: str
    project_name: str
    mode: str = "online"

    _wandb_run: ClassVar[wandb.Run | None] = None

    def get_wandb_run(self) -> wandb.Run:
        if WandBRunResource._wandb_run is None:
            WandBRunResource._wandb_run = wandb.init(
                project=self.project_name,
                name=self.run_name,
                mode=self.mode,  # pyright: ignore[reportArgumentType]
            )
        return WandBRunResource._wandb_run


class WeaveResource(ConfigurableResource):  # pyright: ignore[reportMissingTypeArgument]
    def init_weave(self, run_name: str):
        return weave.init(run_name)


class OCREngineResource(dg.ConfigurableResource[OCREngine]):
    """OCR _engine resource for text and structured extraction."""

    @override
    def create_resource(self, context: dg.InitResourceContext) -> OCREngine:
        ocr_config = cast(
            PytesseractOCRConfig,
            config_manager.load_config_as_model(
                config_name="ocr_model_config",
                config_type=ConfigType.MODELS,
                config_subtype=ModelsConfigSubtype.OCR,
            ),
        )

        return OCREngine.from_config(ocr_config)


class LMv3EngineResource(ConfigurableResource[LMv3Engine]):
    """LayoutLMv3 _engine resource for document understanding."""

    _engine_config: BaseLMv3ModelConfig | None = PrivateAttr(default=None)

    @override
    def setup_for_execution(self, context):
        """Initialize the LLM _engine."""
        self._engine_config = cast(
            BaseLMv3ModelConfig,
            config_manager.load_config_as_model(
                config_name="lmv3_model_config",
                config_type=ConfigType.MODELS,
                config_subtype=ModelsConfigSubtype.LMV3,
            ),
        )

    def get_engine_config(self) -> BaseLMv3ModelConfig:
        """Get the LLM _engine config."""
        if self._engine_config is None:
            raise RuntimeError("LLMEngineConfig not initialized.")
        return self._engine_config

    def get_engine(self, ocr_engine: OCREngine) -> LMv3Engine:
        """Get the LMv3 _engine instance."""
        return LMv3Engine.from_config(
            config=self.get_engine_config(), ocr_engine=ocr_engine
        )


class LLMEngineResource(dg.ConfigurableResource[LLMEngine]):
    """LLM _engine resource for language model operations."""

    _engine_config: LLMEngineConfig | None = PrivateAttr(default=None)

    @override
    def setup_for_execution(self, context):
        """Initialize the LLM _engine."""
        self._engine_config = cast(
            LLMEngineConfig,
            config_manager.load_config_as_model(
                config_name="llm_model_config",
                config_type=ConfigType.MODELS,
                config_subtype=ModelsConfigSubtype.LLM,
            ),
        )

    def get_engine_config(self) -> LLMEngineConfig:
        """Get the LLM _engine config."""
        if self._engine_config is None:
            raise RuntimeError("LLMEngineConfig not initialized.")
        return self._engine_config

    def get_engine(
        self,
        cached: bool = False,
        images_repository: ImageRepository | None = None,
        model_name: str | None = None,
        response_validator: "ResponseValidator[BaseResponse[Any]] | None" = None,
    ) -> LLMEngine | CachedLLMEngine:
        """Get the LLM engine instance.

        Args:
            cached: If True, returns a CachedEngine wrapper for automatic caching.
            images_repository: Required when cached=True for image deduplication.
            model_name: Override the model name from config. If None, uses config default.
            response_validator: for llm engine


        Returns:
            LLMEngine if cached=False, CachedEngine wrapper if cached=True.
        """
        config = self.get_engine_config().model_copy()

        if model_name:
            backend_type = config.backend.type
            config.clients[backend_type].model = model_name

        engine = LLMEngine.from_config(config=config)

        if cached:
            if images_repository is None:
                raise ValueError(
                    "images_repository must be provided when caching is enabled."
                )

            backend, keygen = create_llm_cache_backend(
                engine.used_model, images_repository
            )
            return CachedLLMEngine(
                engine=engine,
                cache_backend=backend,
                key_generator=keygen,
                enabled=True,
                response_validator=response_validator,
            )
        else:
            return engine
