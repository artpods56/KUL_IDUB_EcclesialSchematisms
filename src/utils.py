from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
import random
from typing import List, Dict

import torch
import wandb


def load_labels(dataset: Dataset):
    classes = []
    for example in dataset:
        if "labels" in example.keys():
            if isinstance(example["labels"], list):
                classes.extend(example["labels"])
            else:
                classes.append(example["labels"])
                
    unique_classes = set(classes)
    sorted_classes = sorted(list(unique_classes))

    id2label = {i: label for i, label in enumerate(sorted_classes)}
    label2id = {label: i for i, label in enumerate(sorted_classes)}
    return id2label, label2id, sorted_classes


def prepare_dataset(dataset: Dataset, processor, id2label, label2id, dataset_config):
    
    def prepare_examples(examples):
        images = examples[dataset_config["image_column_name"]]
        words = examples[dataset_config["text_column_name"]]
        boxes = examples[dataset_config["boxes_column_name"]]
        word_labels = examples[dataset_config["label_column_name"]]

        # Since your dataset has string labels, always convert them to IDs
        label_ids = [
            [label2id[label] for label in seq]
            for seq in word_labels
        ]

        encoding = processor(images, words, boxes=boxes, word_labels=label_ids, truncation=True, stride =128, 
                padding="max_length", max_length=512, return_overflowing_tokens=True, return_offsets_mapping=True)  
        offset_mapping = encoding.pop('offset_mapping')
        overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')
        return encoding
    
    


    features = Features({
        'input_ids': Sequence(Value("int64")),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'attention_mask': Sequence(Value("int64")),
        'labels': Sequence(Value("int64")),
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    })

    prepared_dataset = dataset.map(
        prepare_examples,
        batched=True,
        remove_columns=dataset.column_names,
        features=features,
    )
    
    prepared_dataset.set_format("torch")
    
    return prepared_dataset




def _to_fractional(box: List[int]) -> Dict[str, float]:
    """
    LayoutLM* boxes are in the 0-1000 coordinate system.
    WandB defaults to ‘fractional’ domain = values in [0,1].
    """
    min_x, min_y, max_x, max_y = [v / 1000.0 for v in box]
    return {"minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y}

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
    sample_indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    wandb_images = []

    for idx in sample_indices:
        example = dataset[idx]
        image = example["image_pil"]            # PIL image from original dataset
        words = example["words"]                # Words from original dataset
        boxes = example["bboxes"]               # Bounding boxes from original dataset
        labels = example["labels"]              # String labels from original dataset
        
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
            padding="max_length"
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
        
        logits = model(**enc).logits[0]     # sequence_len × num_labels
        preds = logits.argmax(-1).tolist()

        # Build two box lists: predictions & ground-truth
        pred_boxes, gt_boxes = [], []
        # Only iterate over the minimum length to avoid index errors
        min_length = min(len(boxes), len(preds), len(truth))
        
        for i in range(min_length):
            b, p_id, t_id = boxes[i], preds[i], truth[i]
            
            # 1️⃣ predictions
            if p_id != label2id["O"]:
                pred_boxes.append(
                    {
                        "position": _to_fractional(b),
                        "class_id": int(p_id),
                        "box_caption": id2label[p_id],
                        "scores": {"conf": float(torch.softmax(logits[i], -1).max().item())},
                    }
                )
            # 2️⃣ ground truth
            if t_id != label2id["O"]:
                gt_boxes.append(
                    {
                        "position": _to_fractional(b),
                        "class_id": int(t_id),
                        "box_caption": id2label[t_id],
                    }
                )

        wandb_images.append(
            wandb.Image(
                image,
                boxes={
                    "predictions": {
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
# ----------------------------------------------------------------------

def sliding_window(processor, token_boxes, predictions, encoding):
    """
    Process overlapping windows from LayoutLM model to merge tokens and predictions
    based on their spatial positions (bounding boxes).
    
    Args:
        processor: The LayoutLM processor
        token_boxes: List of token bounding boxes in normalized coordinates (0-1000)
        predictions: List of prediction label IDs for each token
        encoding: The model encoding (used for decoding tokens)
    
    Returns:
        boxes: List of unique bounding box coordinates (normalized)
        preds: List of merged predictions (majority vote for each spatial position)
        words: List of merged word strings for each spatial position
    """
    box_token_dict = {}
    for i in range(len(token_boxes)):
        initial_j = 0 if i == 0 else 128  # Skip first 128 tokens for overlapping windows
        for j in range(initial_j, len(token_boxes[i])):
            tb = token_boxes[i][j]
            # skip bad boxes
            if not hasattr(tb, "__len__") or len(tb) != 4 or tb == [0, 0, 0, 0]:
                continue
            # Use normalized coordinates directly as key (more consistent than pixel coords)
            key = tuple(tb)  # Use normalized bbox coordinates as key
            tok = processor.tokenizer.decode(encoding["input_ids"][i][j]).strip()
            box_token_dict.setdefault(key, []).append(tok)

    # build predictions dict with the *same* keys
    box_prediction_dict = {}
    for i in range(len(token_boxes)):
        for j in range(len(token_boxes[i])):
            tb = token_boxes[i][j]
            if not hasattr(tb, "__len__") or len(tb) != 4 or tb == [0, 0, 0, 0]:
                continue
            key = tuple(tb)  # Same key as above
            box_prediction_dict.setdefault(key, []).append(predictions[i][j])

    # Majority vote on predictions for each spatial position
    boxes = list(box_token_dict.keys())
    words = ["".join(ws) for ws in box_token_dict.values()]
    preds = []
    for key, preds_list in box_prediction_dict.items():
        # Simple majority voting - get the most common prediction
        final = max(set(preds_list), key=preds_list.count)
        preds.append(final)

    return boxes, preds, words

def get_device(config=None):
    """
    Returns the device to be used for PyTorch operations.
    If CUDA is available, it returns 'cuda', otherwise 'cpu'.
    """
    if config is not None:
        if config.run.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, but 'cuda' was specified in the config.")
            return "cuda"
        if config.run.device == "cpu":
            return "cpu"
        else:
            raise ValueError(
                f"Unsupported device '{config.run.device}'. Use 'cuda' or 'cpu'."
            )
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"