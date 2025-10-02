import random
from typing import List, Dict # Retaining List, Dict as they might be used by remaining functions or can be cleaned up later if not.
# Removed: from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D (as these are specific to moved functions)
# However, if log_predictions_to_wandb uses any specific classes from `datasets`, this might need adjustment.
# For now, assuming it doesn't or uses them via `data` object's attributes.

import torch
import wandb
from dataset.utils import _to_fractional # Import the moved function

# Moved load_labels, prepare_dataset to src/data/utils.py
# _to_fractional is now imported from data.utils


@torch.no_grad()
def log_predictions_to_wandb(
    model,
    processor,
    dataset,
    id2label,
    label2id,
    num_samples: int = 12,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval().to(device)

    # W&B needs integer keys ➜ string labels
    class_labels = {int(k): v for k, v in id2label.items()}
    sample_indices = random.sample(
        range(len(dataset)), k=min(num_samples, len(dataset)) # `data` is passed in, so `Dataset` type hint not needed here
    )
    wandb_images = []

    for idx in sample_indices:
        example = dataset[idx]
        image = example["image_pil"]  # PIL image from original data
        words = example["words"]  # Words from original data
        boxes = example["bboxes"]  # Bounding boxes from original data
        labels = example["labels"]  # String labels from original data

        # Convert string labels to IDs for truth
        truth = [label2id[label] for label in labels]

        # Process the example for model inference with proper truncation
        enc = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(device)

        # Get the actual sequence length after truncation
        actual_length = enc.input_ids.shape[1]

        # Truncate truth labels to match the processed sequence length
        # The processor may have truncated the input, so we need to match that
        if len(truth) > actual_length:
            truth = truth[:actual_length]
        elif len(truth) < actual_length:
            # Pad with "O" labels if needed
            truth = truth + [label2id["O"]] * (actual_length - len(truth))

        # Truncate boxes and words to match as well
        if len(boxes) > actual_length:
            boxes = boxes[:actual_length]
            words = words[:actual_length]

        logits = model(**enc).logits[0]  # sequence_len × num_labels
        preds = logits.argmax(-1).tolist()

        # Build two box lists: predictions_data & ground-truth
        pred_boxes, gt_boxes = [], []
        # Only iterate over the minimum length to avoid index errors
        min_length = min(len(boxes), len(preds), len(truth))

        for i in range(min_length):
            b, p_id, t_id = boxes[i], preds[i], truth[i]

            # 1️⃣ predictions_data
            if p_id != label2id["O"]:
                pred_boxes.append(
                    {
                        "position": _to_fractional(b), # Use imported _to_fractional
                        "class_id": int(p_id),
                        "box_caption": id2label[p_id],
                        "scores": {
                            "conf": float(torch.softmax(logits[i], -1).max().item())
                        },
                    }
                )
            # 2️⃣ ground truth
            if t_id != label2id["O"]:
                gt_boxes.append(
                    {
                        "position": _to_fractional(b), # Use imported _to_fractional
                        "class_id": int(t_id),
                        "box_caption": id2label[t_id],
                    }
                )

        wandb_images.append(
            wandb.Image(
                image,
                boxes={
                    "predictions_data": {
                        "box_data": pred_boxes,
                        "class_labels": class_labels,
                    },
                    "ground_truth": {
                        "box_data": gt_boxes,
                        "class_labels": class_labels,
                    },
                },
            )
        )

    wandb.log({"📄 sample_pages": wandb_images})

# Removed local _to_fractional_wandb as we are now importing _to_fractional from data.utils


# ----------------------------------------------------------------------


def get_device(config=None):
    """
    Returns the device to be used for PyTorch operations.
    If CUDA is available, it returns 'cuda', otherwise 'cpu'.
    """
    if config is not None:
        if config.run.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available, but 'cuda' was specified in the config."
                )
            return "cuda"
        if config.run.device == "cpu":
            return "cpu"
        else:
            raise ValueError(
                f"Unsupported device '{config.run.device}'. Use 'cuda' or 'cpu'."
            )
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"

