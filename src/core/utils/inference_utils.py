import torch
from structlog import get_logger
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

logger = get_logger(__name__)

def unnormalize_bbox(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def get_model_and_processor(cfg):
    processor = AutoProcessor.from_pretrained(
        cfg.processor.checkpoint, local_files_only=True if cfg.processor.local_files_only else False, apply_ocr=False
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        cfg.inference.checkpoint, local_files_only=True if cfg.inference.local_files_only else False, device_map="auto"
    )
    
    return model, processor

def sliding(processor, token_boxes, predictions, encoding, width, height):
    # for i in range(0, len(token_boxes)):
    #     for j in range(0, len(token_boxes[i])):
    #         print("label is: {}, bbox is: {} and the text is: {}".format(predictions[i][j], token_boxes[i][j],  processor.tokenizer.decode(encoding["input_ids"][i][j]) ))

    box_token_dict = {}
    for i in range(len(token_boxes)):
        initial_j = 0 if i == 0 else 128
        for j in range(initial_j, len(token_boxes[i])):
            tb = token_boxes[i][j]
            # skip bad boxes
            if not hasattr(tb, "__len__") or len(tb) != 4 or tb == [0, 0, 0, 0]:
                continue
            # normalize once here
            unnorm = unnormalize_bbox(tb, width, height)
            key = tuple(round(x, 2) for x in unnorm)  # use a consistent key
            tok = processor.tokenizer.decode(encoding["input_ids"][i][j]).strip()
            box_token_dict.setdefault(key, []).append(tok)

    # build predictions dict with the *same* keys
    box_prediction_dict = {}
    for i in range(len(token_boxes)):
        for j in range(len(token_boxes[i])):
            tb = token_boxes[i][j]
            if not hasattr(tb, "__len__") or len(tb) != 4 or tb == [0, 0, 0, 0]:
                continue
            unnorm = unnormalize_bbox(tb, width, height)
            key = tuple(round(x, 2) for x in unnorm)
            box_prediction_dict.setdefault(key, []).append(predictions[i][j])

    # now your majority‐vote on box_pred_dict → preds
    boxes = list(box_token_dict.keys())
    words = ["".join(ws) for ws in box_token_dict.values()]
    preds = []
    for key, preds_list in box_prediction_dict.items():
        # Simple majority voting - get the most common prediction
        final = max(set(preds_list), key=preds_list.count)
        preds.append(final)

    return boxes, preds, words

@torch.no_grad()
def retrieve_predictions(image, processor, model, words=None, bboxes=None):
    """
    Retrieve predictions for a single example.

    Return:
        boxes
        preds
        flattened_words
    """

    width, height = image.size
    
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

    # Move all input tensors to the same device as the model
    device = next(model.parameters()).device
    for k, v in encoding.items():
        if isinstance(v, torch.Tensor):
            encoding[k] = v.clone().detach().to(device)
        else:
            encoding[k] = v.clone().detach()

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()


    if len(token_boxes) == 512:
        predictions = [predictions]
        token_boxes = [token_boxes]


    boxes, preds, flattened_words = sliding(
        processor, token_boxes, predictions, encoding, width, height
    )

    return boxes, preds, flattened_words




