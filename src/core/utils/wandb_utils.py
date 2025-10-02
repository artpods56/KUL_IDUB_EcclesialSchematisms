import random
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

import wandb
from lmv3.utils.inference_utils import retrieve_predictions
from dataset.utils import _to_fractional # Import the centralized function


# Removed local bbox_to_fractional, using imported _to_fractional instead

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

            ground_truths = example[config.dataset.label_column_name]
            image_pil = example[config.dataset.image_column_name]
            ground_words = example[config.dataset.text_column_name]
            ground_boxes = example[config.dataset.boxes_column_name]

            pred_boxes = []
            gtruth_boxes = []

            bboxes, predictions, words = retrieve_predictions(
                image=image_pil, processor=processor, model=model, words=ground_words, bboxes=ground_boxes
            )

            for ground_truth, bbox, pred, word in zip(
                ground_truths, bboxes, predictions, words
            ):
                label = id2label[pred]

                if label != "O":
                    pred_boxes.append(
                        {
                            "position": _to_fractional(bbox), # Use imported function
                            "class_id": int(pred),
                            "box_caption": id2label[pred],
                        }
                    )
                # Ground truth
                if ground_truth != "O":
                    gtruth_boxes.append(
                        {
                            "position": _to_fractional(bbox), # Use imported function
                            "class_id": int(label2id[ground_truth]),
                            "box_caption": ground_truth,
                        }
                    )

            wandb_images.append(
                wandb.Image(
                    image_pil,
                    boxes={
                        "predictions_data": {
                            "box_data": pred_boxes,
                            "class_labels": id2label,  # Pass the full id2label dict
                        },
                        "ground_truth": {
                            "box_data": gtruth_boxes,
                            "class_labels": id2label,  # Pass the full id2label dict
                        },
                    },
                )
            )

    return wandb_images
