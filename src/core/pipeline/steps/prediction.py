"""
Prediction steps for the pipeline.

This module provides concrete implementations of prediction steps for the pipeline.
It includes steps for OCR, LMv3, and LLM predictions_data.
"""

import re
from typing import List, Optional

from lingua import LanguageDetectorBuilder, Language

from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model
from core.models.ocr.model import OcrModel
from core.pipeline.steps.base import SampleProcessingStep
from core.schemas.data.pipeline import PipelineData
from core.schemas.data.schematism import SchematismPage


class LanguageDetectionStep(SampleProcessingStep[PipelineData, PipelineData]):
    """Processing step for detecting the language of text content.
    
    This step uses the Lingua library to detect the language of OCR-extracted text
    with configurable language restrictions and text preprocessing.
    """

    def __init__(self, languages: List[str], min_sentence_chars: int = 10, max_sentences: Optional[int] = 100, *args,
                 **kwargs):
        """Initialize the language detector with a restricted language set.
        
        Args:
            languages: List of Language enum names (e.g. ["POLISH", "GERMAN"]).
            min_sentence_chars: Minimum character count for sentence processing.
            max_sentences: Maximum number of sentences to process (None for unlimited).
        """
        super().__init__(*args, **kwargs)

        self.languages = self._map_languages(languages)
        self.min_sentence_chars = min_sentence_chars
        self.max_sentences = max_sentences
        if self.languages:
            self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        else:
            # Fallback to all languages if mapping failed for any reason
            self.detector = LanguageDetectorBuilder.from_all_languages().build()

    def _map_languages(self, languages: List[str]) -> List[Language]:
        """Map language strings to Lingua Language enums.
        
        Args:
            languages: List of language code strings.
            
        Returns:
            List of successfully mapped Language enum values.
        """
        mapped_languages = []
        for language in languages:
            try:
                mapped_languages.append(Language.from_str(language))
            except (KeyError, ValueError) as e:
                self.logger.warning("Language not found, ignoring.", language=language, error=e)

        return mapped_languages

    def _preprocess_text(self, text: str) -> str:
        """Clean text: Remove numbers and collapse whitespace without stripping sentence punctuation.
        
        Args:
            text: The input text to preprocess.
            
        Returns:
            The cleaned and preprocessed text.
        """
        # Join hyphenated line breaks common in OCR
        text = re.sub(r"-\s*\n\s*", "", text)
        # Remove digits
        text = re.sub(r"\d+", "", text)
        # Collapse whitespace
        text = " ".join(text.split())
        return text

    def process_sample(self, data: PipelineData, **kwargs) -> PipelineData:
        """Detect language for a single sample with confidence scoring.
        
        Args:
            data: Pipeline data containing text to analyze.
            **kwargs: Additional keyword arguments (unused).
            
        Returns:
            Updated pipeline data with detected language and confidence.
            
        Raises:
            ValueError: If no text is available for language detection.
        """

        if not data.text:
            raise ValueError("No OCR text for language detection, falling back to performing OCR on image.")

        cleaned_text = self._preprocess_text(data.text)

        detections = {detection.language: detection.value
                      for detection in
                      self.detector.compute_language_confidence_values(cleaned_text)}

        language = max(detections, key=lambda k: detections[k])

        data.language = language.name
        data.language_confidence = detections[language]
        return data


class OCRStep(SampleProcessingStep[PipelineData, PipelineData]):
    """Processing step for Optical Character Recognition (OCR).
    
    This step extracts text from images using a configurable OCR model,
    with support for text-only extraction.
    """

    def __init__(self, ocr_model: OcrModel, force_ocr: bool = False, text_only: bool = True, *args, **kwargs):
        """Initialize the OCR step with the specified model and configuration.
        
        Args:
            ocr_model: The OCR model instance to use for text extraction.
            text_only: Whether to extract only text (True) or include bounding boxes (False).
        """
        super().__init__(*args, **kwargs)

        self.ocr_model = ocr_model
        self.force_ocr = force_ocr
        self.text_only = text_only

    def process_sample(self, data: PipelineData, **kwargs) -> PipelineData:
        """Perform OCR on the image in the pipeline data.
        
        Args:
            data: Pipeline data containing the image to process.
            **kwargs: Additional keyword arguments passed to the OCR model.
            
        Returns:
            Updated pipeline data with extracted text.
            
        Raises:
            ValueError: If no image is provided and no text is available.
            NotImplementedError: If text_only is False (not yet supported).
        """
        if data.image is None:
            if data.text:
                self.logger.warning("OCR text provided, skipping OCR.")
                return data
            else:
                raise ValueError("No image provided for OCR and no text available.")

        if not self.text_only:
            raise NotImplementedError("This step supports text only prediction.")

        if data.text is not None and not self.force_ocr:
            return data

        data.text = self.ocr_model.predict(
            image=data.image,
            text_only=True,
            **kwargs
        )
        return data


class LMv3PredictionStep(SampleProcessingStep[PipelineData, PipelineData]):
    """Processing step for LayoutLMv3 model predictions_data.
    
    This step performs structured document understanding using a LayoutLMv3 model,
    extracting structured information from document images.
    """

    def __init__(self, lmv3_model: LMv3Model, *args, **kwargs):
        """Initialize the LMv3 prediction step with the specified model.
        
        Args:
            lmv3_model: The LayoutLMv3 model instance to use for predictions_data.
        """
        super().__init__(*args, **kwargs)
        self.lmv3_model = lmv3_model

    def process_sample(self, data: PipelineData, **kwargs) -> PipelineData:
        """Generate structured predictions_data using the LayoutLMv3 model.
        
        Args:
            data: Pipeline data containing the image to analyze.
            **kwargs: Additional keyword arguments passed to the model.
            
        Returns:
            Updated pipeline data with LMv3 predictions_data.
        """
        if data.image is None:
            self.logger.warning("No image provided for LMv3 prediction, skipping.")
            return data

        structured_predictions = self.lmv3_model.predict(
            data.image,
            raw_predictions=False,
            **kwargs
        )

        try:
            data.lmv3_prediction = SchematismPage(**structured_predictions)  # type: ignore[arg-type]
        except Exception as e:
            self.logger.warning("Failed to coerce LMv3 predictions_data to SchematismPage. Continuing without predictions_data.",
                                error=e)

        return data


class LLMPredictionStep(SampleProcessingStep[PipelineData, PipelineData]):
    """Processing step for Large Language Model (LLM) predictions_data.
    
    This step generates structured predictions_data using a large language model,
    optionally incorporating hints from previous LayoutLMv3 predictions_data.
    """

    def __init__(self, llm_model: LLMModel, system_prompt:str, user_prompt: str, use_lmv3_hints: bool = True, use_ground_truth: bool = False, *args, **kwargs):
        """Initialize the LLM prediction step with the specified model and configuration.
        
        Args:
            llm_model: The LLM model instance to use for predictions_data.
            use_lmv3_hints: Whether to use LMv3 predictions_data as hints for the LLM.
        """
        super().__init__(*args, **kwargs)
        self.llm_model = llm_model
        self.use_lmv3_hints = use_lmv3_hints
        self.use_ground_truth = use_ground_truth
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def process_sample(self, data: PipelineData, **kwargs) -> PipelineData:
        """Generate structured predictions_data using the Large Language Model.
        
        Args:
            data: Pipeline data containing image and/or text to analyze.
            **kwargs: Additional keyword arguments (unused).
            
        Returns:
            Updated pipeline data with LLM predictions_data.
            
        Raises:
            ValueError: If neither image nor text is provided, or if prediction coercion fails.
        """
        if data.image is None and data.text is None:
            raise ValueError("At least one of image or text must be provided for LLM processing.")

        context = {}

        if self.use_lmv3_hints and data.lmv3_prediction is not None:
            context["hints"] = data.lmv3_prediction.model_dump()


        if self.use_ground_truth and data.ground_truth is not None:
            context["ground_truth"] = data.ground_truth.model_dump()



        try:
            response, parsed_messages = self.llm_model.predict(
                image=data.image,
                text=data.text,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                context=context,
            )

            data.parsed_messages = parsed_messages
            data.llm_prediction = SchematismPage(**response)
        except Exception as e:

            self.logger.error("Failed to coerce LLM predictions_data to SchematismPage.", error=e)
            raise ValueError("Failed to coerce LLM prediction to SchematismPage") from e
        return data
