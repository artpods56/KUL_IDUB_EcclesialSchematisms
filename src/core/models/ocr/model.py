import numpy as np
from typing import List, Tuple, Union, Optional, cast

from PIL import Image
import pytesseract
from structlog import get_logger

from core.caches.ocr_cache import PyTesseractCache
from core.schemas.caches.entries import PyTesseractCacheItem

from core.caches.utils import get_image_hash

def ocr_page(
    pil_image: Image.Image,
    language: str = "lat+pol+rus",
    text_only: bool = False,
) -> Union[str, Tuple[List[str], List[List[int]]]]:
    """Run PyTesseract OCR on a page.

    Args:
        pil_image: The image to be processed.
        language: Languages passed to Tesseract (e.g. "eng+deu").
        text_only: If ``True`` only return the full text string (no word bboxes).

    Returns:
        Either a string (``text_only=True``) or a tuple ``(words, bboxes)``.
        Bounding boxes follow LayoutLMv3 convention of 0-1000 normalized coordinates:
        ``[xmin, ymin, xmax, ymax]``.
    """

    if text_only:
        return pytesseract.image_to_string(
            np.array(pil_image.convert("L")),
            lang=language,
            config="--psm 6 --oem 3",
        )

    width, height = pil_image.size

    ocr_dict = pytesseract.image_to_data(
        np.array(pil_image.convert("L")),
        output_type=pytesseract.Output.DICT,
        lang=language,
        config="--psm 6 --oem 3",
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


class OcrModel:
    """PyTesseract OCR model wrapper with caching.

    The model exposes a unified ``predict`` method returning either the full OCR text or
    word-level bounding boxes depending on the *text_only* flag.
    """

    def __init__(
        self,
        config = None,
        enable_cache: bool = True,
        language: str = "lat+pol+rus",
    ) -> None:
        self.config = config
        self.language = language

        self.logger = get_logger(__name__).bind(language=language)

        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = PyTesseractCache(language=language)

    def _predict(self, pil_image: Image.Image, text_only: bool = False):
        return ocr_page(pil_image, language=self.language, text_only=text_only)


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