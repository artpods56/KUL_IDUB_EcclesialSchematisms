import torch


@torch.no_grad()
def retrieve_predictions(image, processor, model, words=None, bboxes=None):
    """
    Retrieve predictions for a single example.

    Return:
        boxes
        preds
        flattened_words
    """
    if words is not None and bboxes is not None:
        encoding = processor(
            image,
            words=words,
            boxes=bboxes,
            apply_ocr=False,
            truncation=True,
            stride=128,
            padding="max_length",
            max_length=512,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

    else:
        encoding = processor(
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
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    if len(token_boxes) == 512:
        predictions = [predictions]
        token_boxes = [token_boxes]

    boxes, preds, flattened_words = sliding_window(
        processor, token_boxes, predictions, encoding
    )

    return boxes, preds, flattened_words


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
        initial_j = (
            0 if i == 0 else 128
        )  # Skip first 128 tokens for overlapping windows
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
