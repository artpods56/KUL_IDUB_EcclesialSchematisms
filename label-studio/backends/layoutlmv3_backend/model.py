import logging
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import boto3
import requests
import torch
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import (
    ModelResponse,
    PredictionValue,
    SingleTaskPredictions,
)
from PIL import Image
from torch._prims_common import DeviceLikeType, check
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

from utils import merge_bio_entities, pixel_bbox_to_percent, sliding_window

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from label_studio_ml.utils import get_single_tag_keys

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


MODEL_DIR = "/app/data/models"
checkpoint_path = os.path.join(
    MODEL_DIR, os.getenv("MODEL_CHECKPOINT", "checkpoint-400")
)

logger.info(f"Loading model from {checkpoint_path}")
_GLOBAL_MODEL = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint_path)
logger.info("Loading processor")
_GLOBAL_PROCESSOR = AutoProcessor.from_pretrained(
    "microsoft/layoutlmv3-large", apply_ocr=True
)


@dataclass
class CheckpointLoadError(RuntimeError):
    def __init__(self, message: str, checkpoint_path: str):
        self.message = message
        self.checkpoint_path = checkpoint_path

    def __str__(self) -> str:
        return f"Could not load model checkpoint from {self.checkpoint_path}: {self.message}"


class LayoutLMv3Backend(LabelStudioMLBase):
    """Label‑Studio backend with training & inference for LayoutLMv3."""

    MODEL_DIR = "/app/data/models"
    model_checkpoint = os.getenv("MODEL_CHECKPOINT", "checkpoint-400")
    _model: LayoutLMv3ForTokenClassification = None  # type: ignore

    def __init__(self, **kwargs):
        super(LayoutLMv3Backend, self).__init__(**kwargs)

        self._lazy_init()

        endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name = os.getenv("AWS_REGION", "us-east-1")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        params = get_single_tag_keys(
            self.parsed_label_config, "RectangleLabels", "Image"
        )

        self.from_name, self.to_name, self.value, self.labels_in_config = params

        self.label_map = self._build_labels_from_model_config()

    def _build_labels_from_model_config(self) -> Dict[int, str]:

        label_map = {
            v: k[2:] for k, v in self._model.config.label2id.items() if k != "O"
        }

        logger.info(f"Class labels supported by this model checkpoint: {label_map}")

        filtered_label_map = {
            key: value
            for key, value in label_map.items()
            if value
            in self.labels_in_config  # Only keep labels defined in the project config
        }

        logger.info(
            f"Filtered label map (aligned with project config): {filtered_label_map}"
        )

        return filtered_label_map

    def _lazy_init(self):
        if not self._model:
            checkpoint_path = Path(os.path.join(self.MODEL_DIR, self.model_checkpoint))
            try:
                logger.info("Loading the model from cache.")
                # self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                #     checkpoint_path
                # )
                self._model = _GLOBAL_MODEL
                self._processor = _GLOBAL_PROCESSOR
                # self._processor = AutoProcessor.from_pretrained(
                #     "microsoft/layoutlmv3-large", apply_ocr=True
                # )
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"Could not load checkpoint from {checkpoint_path} as {e}")
                raise CheckpointLoadError(str(e), str(checkpoint_path))

    def _load_image(self, uri: str) -> Image.Image:
        """
        Wczytuje obraz z S3/HTTP bez zapisu na dysk.
        """
        if uri.startswith("s3://"):
            # s3://bucket/key
            _, bucket_key = uri.split("s3://", 1)
            bucket, key = bucket_key.split("/", 1)
            obj = self.s3.get_object(Bucket=bucket, Key=key)
            buf = BytesIO(obj["Body"].read())
            return Image.open(buf).convert("RGB")

        # fallback na HTTP/HTTPS
        resp = requests.get(uri, stream=True)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")

    def setup(self):
        self.set("model_version", f"{self.__class__.__name__}-v0.0.1")

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        logger.info("Starting prediction on %d tasks", len(tasks))
        all_task_predictions = []

        for task_idx, task in enumerate(tasks):
            image_uri = task["data"]["image"]
            image = self._load_image(image_uri)
            image_width, image_height = image.size

            encoding = self._processor(
                image,
                truncation=True,
                stride=128,
                padding="max_length",
                max_length=512,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            offset_mapping = encoding.pop("offset_mapping")
            overflow_to_sample_mapping = encoding.pop("overflow_to_sample_mapping")
            x = []
            for i in range(0, len(encoding["pixel_values"])):
                ndarray_pixel_values = encoding["pixel_values"][i]
                tensor_pixel_values = torch.tensor(ndarray_pixel_values)
                x.append(tensor_pixel_values)

            x = torch.stack(x)
            encoding["pixel_values"] = x

            for k, v in encoding.items():
                encoding[k] = torch.tensor(v)

            with torch.no_grad():
                outputs = self._model(**encoding)

            logits = outputs.logits

            predicted_tokens = logits.argmax(-1).squeeze().tolist()
            token_bboxes = encoding.bbox.squeeze().tolist()

            if len(token_bboxes) == 512:
                predicted_tokens = [predicted_tokens]
                token_bboxes = [token_bboxes]

            bboxes, predictions, words = sliding_window(
                self._processor,
                token_bboxes,
                predicted_tokens,
                encoding,
                image_width,
                image_height,
            )

            # merge BIO entities

            merged_bboxes, merged_sentences, merged_classes = merge_bio_entities(
                bboxes,
                predictions,
                words,
                self._model.config.id2label,
                o_label_id=14,
                verbose=True,
            )
            logger.info(f"Task {task_idx} - Merged sentences: {merged_sentences}")
            logger.info(f"Merged classes: {merged_classes}")
            logger.info(f"Task {task_idx} - Merged bboxes: {merged_bboxes}")

            logger.info(f"Merged classes: {merged_classes}")
            logger.info(f"Label map: {self.label_map}")

            current_task_results = []
            for bbox, prediction_id in zip(merged_bboxes, merged_classes):

                if prediction_id in self.label_map:
                    x_percent, y_percent, width_percent, height_percent = (
                        pixel_bbox_to_percent(
                            bbox=bbox,
                            image_width=image_width,
                            image_height=image_height,
                        )
                    )
                    label = self.label_map[prediction_id]
                    result_item = {
                        "type": "rectanglelabels",
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "original_width": image_width,
                        "original_height": image_height,
                        "image_rotation": 0,
                        "value": {
                            "rectanglelabels": [label],
                            "x": x_percent,
                            "y": y_percent,
                            "width": width_percent,
                            "height": height_percent,
                        },
                        "score": 1.0,
                    }
                    current_task_results.append(result_item)

            task_prediction = {"result": current_task_results}
            all_task_predictions.append(task_prediction)

        return ModelResponse(
            predictions=all_task_predictions,
            model_version=self.get("model_version") or "unknown",
        )
