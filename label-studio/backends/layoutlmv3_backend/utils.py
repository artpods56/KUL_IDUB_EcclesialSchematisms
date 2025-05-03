import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def unnormalize_bbox(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def pixel_bbox_to_percent(
    bbox: Tuple[int, int, int, int], image_width: int, image_height: int
) -> Tuple[float, float, float, float]:
    """
    Zamienia bbox w pikselach (x1,y1,x2,y2)
    na wartości procentowe (x%, y%, width%, height%).
    """
    x1, y1, x2, y2 = bbox
    x_pct: float = (x1 / image_width) * 100.0
    y_pct: float = (y1 / image_height) * 100.0
    width_pct: float = ((x2 - x1) / image_width) * 100.0
    height_pct: float = ((y2 - y1) / image_height) * 100.0
    return x_pct, y_pct, width_pct, height_pct


def sliding_window(processor, token_boxes, predictions, encoding, width, height):
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


def merge_bio_entities(
    bboxes, predictions, tokens, id2label, o_label_id=14, verbose=False
):
    """
    Merge consecutive BIO entities into single bounding boxes and tokens.

    Args:
        boxes (list): List of bounding boxes [x1, y1, x2, y2].
        preds (list): List of predicted label IDs.
        words (list): List of tokens/words.
        id2label (dict): Mapping from label ID to label string (e.g., {0: 'B-parish', ...}).
        o_label_id (int): The label ID for 'O' (outside) class.
        verbose (bool): If True, print debug info.

    Returns:
        merged_boxes (list): List of merged bounding boxes.
        merged_tokens (list): List of merged entity strings.
        merged_classes (list): List of merged entity class names.
    """
    merged_boxes = []
    merged_tokens = []
    merged_classes = []

    previous_class_name = None
    bbox_stack = []
    token_stack = []

    def merge_entity(bbox_stack, token_stack):
        merged_bbox = [
            min(bbox[0] for bbox in bbox_stack),
            min(bbox[1] for bbox in bbox_stack),
            max(bbox[2] for bbox in bbox_stack),
            max(bbox[3] for bbox in bbox_stack),
        ]
        merged_token = " ".join(token_stack)
        return merged_bbox, merged_token

    for bbox, prediction, token in zip(bboxes, predictions, tokens):
        if prediction == o_label_id:
            continue

        # initial case
        if previous_class_name is None:
            previous_class_name = prediction
            bbox_stack = [bbox]
            token_stack = [token]

        elif previous_class_name == prediction:

            if id2label[prediction][:2] == "B-":
                # this means we have a new entity
                merged_bbox, merged_token = merge_entity(bbox_stack, token_stack)
                merged_boxes.append(merged_bbox)
                merged_tokens.append(merged_token)
                merged_classes.append(previous_class_name)
                previous_class_name = prediction
                bbox_stack = [bbox]
                token_stack = [token]
            elif id2label[prediction][:2] == "I-":
                # this means we have a continuation of the same entity
                previous_class_name = prediction
                bbox_stack.append(bbox)
                token_stack.append(token)

        elif previous_class_name != prediction:
            # this means we have a new entity
            if id2label[prediction][:2] == "I-":
                # this means we have a continuation of the same entity

                previous_class_name = prediction
                bbox_stack.append(bbox)
                token_stack.append(token)
            elif id2label[prediction][:2] == "B-":

                merged_bbox, merged_token = merge_entity(bbox_stack, token_stack)
                merged_boxes.append(merged_bbox)
                merged_tokens.append(merged_token)
                merged_classes.append(previous_class_name)

                previous_class_name = prediction
                bbox_stack = [bbox]
                token_stack = [token]

    if bbox_stack and token_stack:
        merged_bbox, merged_token = merge_entity(bbox_stack, token_stack)
        merged_boxes.append(merged_bbox)
        merged_tokens.append(merged_token)
        merged_classes.append(previous_class_name)

    return merged_boxes, merged_tokens, merged_classes
