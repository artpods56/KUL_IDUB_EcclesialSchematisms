import torch
from utils import sliding_window


@torch.no_grad()
def retrieve_predictions(image, processor, model):
    """
    Retrieve predictions for a single example.
    """

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
