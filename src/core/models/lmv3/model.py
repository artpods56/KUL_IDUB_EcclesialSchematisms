from logging import getLogger
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image
import pytesseract


from core.caches.lmv3_cache import LMv3Cache, LMv3CacheItem
from data.parsing import build_page_json
from core.utils.inference_utils import get_model_and_processor, retrieve_predictions
from core.utils.logging import setup_logging

setup_logging()
logger = getLogger(__name__)

class LMv3Model:
    """LayoutLMv3 model wrapper with unified predict interface and caching."""
    
    def __init__(self, config, enable_cache: bool = True):
        self.config = config
        self.model, self.processor = get_model_and_processor(config)
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = LMv3Cache()

    def ocr_page(self, pil_image: Image.Image):
        """Extract OCR text and bounding boxes from PIL image."""
        width, height = pil_image.size
        
        ocr = pytesseract.image_to_data(
            np.array(pil_image.convert("L")),
            output_type=pytesseract.Output.DICT,
            lang="lat+pol+rus",
            config="--psm 6 --oem 3",
        )
        
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

    def _predict(self, pil_image: Image.Image, raw_preds: bool = False) -> Union[Dict, Tuple[List, List, List]]:
        """Predict on PIL image and return JSON results with caching.
        
        Args:
            pil_image: PIL Image object
            raw_preds: If True, return raw predictions (words, bboxes, preds)
            """

        words, bboxes = self.ocr_page(pil_image)

        grayscale_image = pil_image.convert("L").convert("RGB")
        
        pred_bboxes, pred_ids, _ = retrieve_predictions(
            image=grayscale_image,
            processor=self.processor,
            model=self.model,
            words=words,
            bboxes=bboxes,
        )

        preds = [self.id2label[p] for p in pred_ids]

        return words, pred_bboxes, preds


    def predict(self, pil_image: Image.Image, raw_preds: bool = False) -> Union[Dict, Tuple[List, List, List]]:
        """Predict on PIL image and return JSON results with caching.
        
        Args:
            pil_image: PIL Image object
            raw_preds: If True, return raw predictions (words, bboxes, preds)
            
        Returns:
            Dictionary with structured prediction results or tuple of raw predictions
        """
        if self.enable_cache:
            logger.debug(f"Cache enabled")
            hash_key = self.cache.generate_hash(
                image_hash=self.cache.get_image_hash(pil_image),
                raw_preds=raw_preds
            )
            
            try:
                cached_result = self.cache[hash_key]
            except KeyError:
                cached_result = None

            if cached_result:
                if raw_preds:
                    if cached_result["raw_preds"] is not None:
                        return cached_result["raw_preds"]
                else:
                    if cached_result["structured_preds"] is not None:
                        return cached_result["structured_preds"]
            
            
            words, bboxes, preds = self._predict(pil_image, raw_preds)
            structured_preds = build_page_json(words=words, bboxes=bboxes, labels=preds)

            self.cache[hash_key] = LMv3CacheItem(
                raw_preds=(words, bboxes, preds), 
                structured_preds=structured_preds
                ).model_dump()

            self.cache.save_cache()

            if raw_preds:
                return words, bboxes, preds
            else:
                return structured_preds
        else:
            words, bboxes, preds = self._predict(pil_image, raw_preds)
            
            if raw_preds:
                return words, bboxes, preds
            else:
                return build_page_json(words=words, bboxes=bboxes, labels=preds)
    