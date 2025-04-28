from typing import List, Tuple


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
