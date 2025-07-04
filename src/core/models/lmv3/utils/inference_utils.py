import torch
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

def get_model_and_processor(cfg):
    processor = AutoProcessor.from_pretrained(
        cfg.processor.checkpoint, apply_ocr=False
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        cfg.model.checkpoint,
    )
    
    return model, processor

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
            words,
            boxes=bboxes,
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
        tensor_pixel_values = ndarray_pixel_values.clone().detach()
        x.append(tensor_pixel_values)

    x = torch.stack(x)
    encoding["pixel_values"] = x

    for k, v in encoding.items():
        encoding[k] = v.clone().detach()

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
    box_tokens = {}
    for i in range(len(token_boxes)):
        start = 0 if i == 0 else 128
        for j in range(start, len(token_boxes[i])):
            tb = token_boxes[i][j]
            if len(tb) != 4 or tb == [0, 0, 0, 0]:
                continue
            key = tuple(tb)
            tok = processor.tokenizer.convert_ids_to_tokens(
                int(encoding["input_ids"][i][j])
            )
            box_tokens.setdefault(key, []).append(tok)

    # ► prawidłowa rekonstrukcja wyrazów
    words = [
        processor.tokenizer.convert_tokens_to_string(toks).strip()
        for toks in box_tokens.values()
    ]

    # majority-vote na labelach tak jak było
    box_pred = {}
    for i in range(len(token_boxes)):
        for j, tb in enumerate(token_boxes[i]):
            if len(tb) != 4 or tb == [0, 0, 0, 0]:
                continue
            key = tuple(tb)
            box_pred.setdefault(key, []).append(predictions[i][j])

    preds = [max(set(p), key=p.count) for p in box_pred.values()]
    boxes = list(box_tokens.keys())
    return boxes, preds, words

