import numpy as np
from typing import List, Literal, Tuple, Union, Optional, cast, overload

from PIL import Image
from omegaconf import DictConfig
import pytesseract
from structlog import get_logger

from core.caches.ocr_cache import PyTesseractCache
from core.schemas.data.cache import PyTesseractCacheItem

from core.caches.utils import get_image_hash

from core.models.base import ConfigurableModel

def ocr_page(
    pil_image: Image.Image,
    language: str = "lat+pol+rus",
    text_only: bool = False,
    psm_mode: int = 6,
    oem_mode: int = 3,
) -> Union[str, Tuple[List[str], List[List[int]]]]:
    """Run PyTesseract OCR on a page.

    Args:
        pil_image: The image to be processed.
        language: Languages passed to Tesseract (e.g. "eng+deu").
        text_only: If ``True`` only return the full text string (no word bboxes).
        psm_mode: Page segmentation mode for Tesseract.
        oem_mode: OCR Engine Mode for Tesseract.

    Returns:
        Either a string (``text_only=True``) or a tuple ``(words, bboxes)``.
        Bounding boxes follow LayoutLMv3 convention of 0-1000 normalized coordinates:
        ``[xmin, ymin, xmax, ymax]``.
    """
    config = f"--psm {psm_mode} --oem {oem_mode}"

    if text_only:
        return pytesseract.image_to_string(
            np.array(pil_image.convert("L")),
            lang=language,
            config=config,
        )

    width, height = pil_image.size

    ocr_dict = pytesseract.image_to_data(
        np.array(pil_image.convert("L")),
        output_type=pytesseract.Output.DICT,
        lang=language,
        config=config,
    )

    words: List[str] = []
    bboxes: List[List[int]] = []

    for i, word in enumerate(ocr_dict["text"]):
        # Level 5 corresponds to word level
        if ocr_dict["level"][i] != 5:
            continue

        w = word.strip()
        if not w or int(ocr_dict["conf"][i]) < 0:
            continue

        xmin, ymin = ocr_dict["left"][i], ocr_dict["top"][i]
        xmax = xmin + ocr_dict["width"][i]
        ymax = ymin + ocr_dict["height"][i]

        box = [
            int(1000 * xmin / width),
            int(1000 * ymin / height),
            int(1000 * xmax / width),
            int(1000 * ymax / height),
        ]

        words.append(w)
        bboxes.append(box)

    return words, bboxes


class OcrModel(ConfigurableModel):
    """PyTesseract OCR model wrapper with caching.

    The model exposes a unified ``predict`` method returning either the full OCR text or
    word-level bounding boxes depending on the *text_only* flag.
    """

    def __init__(
        self,
        config: DictConfig,
        enable_cache: Optional[bool] = None,
        language: Optional[str] = None,
    ) -> None:
        self.language = language or config.get("language", "lat+pol+rus")
        self.enable_cache = enable_cache if enable_cache is not None else config.get("enable_cache", True)
        self.psm_mode = config.get("psm_mode", 6)
        self.oem_mode = config.get("oem_mode", 3)

        self.logger = get_logger(__name__).bind(language=self.language)

        if self.enable_cache:
            self.cache = PyTesseractCache(language=self.language)

    @classmethod
    def from_config(cls, config: DictConfig) -> "OcrModel":
        return cls(config=config)

    def _predict(self, pil_image: Image.Image, text_only: bool = False):
        return ocr_page(
            pil_image, 
            language=self.language, 
            text_only=text_only,
            psm_mode=self.psm_mode,
            oem_mode=self.oem_mode
        )


    @overload
    def predict(self, image: Image.Image, text_only: Literal[True], **kwargs) -> str:
        """Perform OCR on *image* and return text only."""

    @overload
    def predict(self, image: Image.Image, text_only: Literal[False], **kwargs) -> Tuple[list, list]:
        """Perform OCR on *image* and return words and bounding boxes."""

    def predict(
        self,
        image: Image.Image,
        text_only: bool = False,
        **kwargs,
    ) -> Union[str, Tuple[list, list]]:
        """Perform OCR on *image*.

        Args:
            image: Input image as ``PIL.Image``.
            text_only: If ``True`` returns a single string of the page; otherwise returns
                a tuple ``(words, bboxes)``.
            metadata: Optional metadata for caching and tracking
        """

        if not self.enable_cache:
            return self._predict(image, text_only=text_only)

        else:
            image_hash = get_image_hash(image)
            hash_key = self.cache.generate_hash(image_hash=image_hash)

            try:
                cache_item = cast(
                    dict,
                    self.cache.get(key=hash_key)
                    )

                if cache_item is not None:
                    if text_only:
                        return cache_item["text"]
                    else:
                        return cache_item["words"], cache_item["bbox"]
            except KeyError:
                pass

            text = cast(str, self._predict(image, text_only=True))
            words, bboxes = cast(Tuple[list, list], self._predict(image, text_only=False))

            cache_item_data = {
                "text": text,
                "bbox": bboxes,
                "words": words,
            }

            schematism = kwargs.get("schematism", None)
            filename = kwargs.get("filename", None)

            cache_item = PyTesseractCacheItem(**cache_item_data)
            self.cache.set(
                key=hash_key,
                value=cache_item.model_dump(),
                schematism=schematism,
                filename=filename,
            )

            if text_only:
                return text
            else:
                return words, bboxes