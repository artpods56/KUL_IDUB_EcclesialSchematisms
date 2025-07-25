from typing import Dict, List, Tuple, Union, Optional, cast

import numpy as np
from PIL import Image
import pytesseract
from pydantic_core import ValidationError
from structlog import get_logger

from core.data.parsing import build_page_json, repair_bio_labels
from core.utils.inference_utils import get_model_and_processor, retrieve_predictions
from core.schemas.caches.entries import LMv3CacheItem
from core.caches.lmv3_cache import LMv3Cache
from core.caches.utils import get_image_hash

logger = get_logger(__name__)


def ocr_page(pil_image: Image.Image, text_only: bool = False) -> Union[Tuple[List, List], str]:
    """Extract OCR text and bounding boxes from PIL image."""
    width, height = pil_image.size

    if not text_only:
        ocr = pytesseract.image_to_data(
            np.array(pil_image.convert("L")),
            output_type=pytesseract.Output.DICT,
            lang="lat+pol+rus",
            config="--psm 6 --oem 3",
        )
    else:
        ocr = pytesseract.image_to_string(
            np.array(pil_image.convert("L")),
            lang="lat+pol+rus",
            config="--psm 6 --oem 3",
        )
        return ocr

    words, bboxes = [], []
    for i, word in enumerate(ocr["text"]):
        if ocr["level"][i] != 5:
            continue
        if not (w := word.strip()) or int(ocr["conf"][i]) < 0:
            continue
        xmin, ymin = ocr["left"][i], ocr["top"][i]
        xmax, ymax = xmin + ocr["width"][i], ymin + ocr["height"][i]

        box = [
            int(1000 * xmin / width),
            int(1000 * ymin / height),
            int(1000 * xmax / width),
            int(1000 * ymax / height),
        ]

        bboxes.append(box)
        words.append(w)

    return words, bboxes


class LMv3Model:
    """LayoutLMv3 model wrapper with unified predict interface and caching."""
    
    def __init__(self, config, enable_cache: bool = True):
        self.config = config
        self.model, self.processor = get_model_and_processor(config)
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = LMv3Cache(
                checkpoint=config.inference.checkpoint,
            )

    def _predict(self, pil_image: Image.Image,) -> Tuple[List, List, List]:
        """Predict on PIL image and return JSON results with caching.
        
        Args:
            pil_image: PIL Image object
            """

        words, bboxes = cast(
            Tuple[list,list],
            ocr_page(pil_image)
        )

        grayscale_image = pil_image.convert("L").convert("RGB")
        
        prediction_bboxes, prediction_ids, _ = retrieve_predictions(
            image=grayscale_image,
            processor=self.processor,
            model=self.model,
            words=words,
            bboxes=bboxes,
        )

        predictions = [self.id2label[p] for p in prediction_ids]

        return words, prediction_bboxes, predictions


    def predict(self, pil_image: Image.Image, raw_predictions: bool = False, **kwargs) -> Union[Dict, Tuple[List, List, List]]:
        """Predict on PIL image and return JSON results with caching.
        
        Args:
            pil_image: PIL Image object
            raw_predictions: If True, return raw predictions (words, bboxes, preds)
        Returns:
            Dictionary with structured prediction results or tuple of raw predictions
        """
        if self.enable_cache:
            hash_key = self.cache.generate_hash(
                image_hash=get_image_hash(pil_image),
                raw_predictions=raw_predictions
            )
            
            try:
                cache_item_data = cast(
                    Optional[dict], self.cache.get(key=hash_key)
                )
                if cache_item_data is not None:
                    cache_item = LMv3CacheItem(**cache_item_data)

                    if raw_predictions:
                        return cache_item.raw_predictions
                    else:
                        return cache_item.structured_predictions.model_dump()

            except ValidationError as e:
                self.cache.delete(key=hash_key)

            words, bboxes, predictions = self._predict(pil_image)
            repaired_predictions = repair_bio_labels(predictions)

            structured_predictions = build_page_json(words=words, bboxes=bboxes, labels=repaired_predictions)

            cache_item_data = {
                "raw_predictions": (words, bboxes, predictions),
                "structured_predictions": structured_predictions
            }

            schematism = kwargs.get("schematism", None)
            filename = kwargs.get("filename", None)

            self.cache.set(
                key=hash_key,
                value=LMv3CacheItem(**cache_item_data).model_dump(),
                schematism=schematism,
                filename=filename,
            )

            if raw_predictions:
                return words, bboxes, predictions
            else:
                return structured_predictions
        else:
            words, bboxes, predictions = self._predict(pil_image)
            repaired_predictions = repair_bio_labels(predictions)
            if raw_predictions:
                return words, bboxes, repaired_predictions
            else:
                return build_page_json(words=words, bboxes=bboxes, labels=repaired_predictions)
    