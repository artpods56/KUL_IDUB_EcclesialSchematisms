import random
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

import wandb
from lmv3.utils.inference_utils import retrieve_predictions


def bbox_to_fractional(box: List[int]) -> Dict[str, float]:
    """
    LayoutLM* boxes are in the 0-1000 coordinate system.
    WandB defaults to ‘fractional’ domain = values in [0,1].
    """
    min_x, min_y, max_x, max_y = [v / 1000.0 for v in box]
    return {"minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y}


def log_predictions_to_wandb(
    model: LayoutLMv3ForTokenClassification,
    processor: AutoProcessor,
    datasets_splits: List[Dataset],
    config,
):

    id2label = model.config.id2label
    label2id = model.config.label2id

    wandb_images = []

    for dataset_split in datasets_splits:
        sample_indices = random.sample(
            range(len(dataset_split)),
            k=min(config.wandb.num_prediction_samples, len(dataset_split)),
        )

        for sample_idx in sample_indices:
            example = dataset_split[sample_idx]

            ground_truths = example["labels"]
            image_pil = example[config.dataset.image_column_name]

            pred_boxes = []
            gtruth_boxes = []
            bboxes, predictions, words = retrieve_predictions(
                image=image_pil, processor=processor, model=model
            )

            for ground_truth, bbox, pred, word in zip(
                ground_truths, bboxes, predictions, words
            ):
                label = id2label[pred]

                if label != "O":
                    pred_boxes.append(
                        {
                            "position": bbox_to_fractional(bbox),
                            "class_id": int(pred),
                            "box_caption": id2label[pred],
                        }
                    )
                # 2️⃣ ground truth
                if ground_truth != "O":
                    gtruth_boxes.append(
                        {
                            "position": bbox_to_fractional(bbox),
                            "class_id": int(label2id[ground_truth]),
                            "box_caption": ground_truth,
                        }
                    )

            wandb_images.append(
                wandb.Image(
                    image_pil,
                    boxes={
                        "predictions": {
                            "box_data": pred_boxes,
                            "class_labels": predictions,
                        },
                        "ground_truth": {
                            "box_data": gtruth_boxes,
                            "class_labels": ground_truths,
                        },
                    },
                )
            )

    return wandb_images
