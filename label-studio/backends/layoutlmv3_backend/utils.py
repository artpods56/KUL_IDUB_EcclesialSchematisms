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


import numpy as np # Make sure numpy is imported

def merge_bio_entities(bboxes, predictions, tokens, id2label, o_label_id=14, verbose=False):
    """
    Merge consecutive BIO entities into single bounding boxes and tokens,
    avoiding merging across apparent line breaks or large gaps.

    Args:
        bboxes (list): List of bounding boxes [x1, y1, x2, y2].
        predictions (list): List of predicted label IDs.
        tokens (list): List of tokens/words.
        id2label (dict): Mapping from label ID to label string (e.g., {0: 'B-parish', ...}).
        o_label_id (int): The label ID for 'O' (outside) class.
        verbose (bool): If True, print debug info.

    Returns:
        merged_boxes (list): List of merged bounding boxes.
        merged_tokens (list): List of merged entity strings.
        merged_classes (list): List of merged entity class names (e.g., 'parish').
    """
    # Validate inputs
    if not (len(bboxes) == len(predictions) == len(tokens)):
        raise ValueError("bboxes, predictions, and tokens must have the same length")

    merged_boxes = []
    merged_tokens = []
    merged_classes = []

    bbox_stack = []
    token_stack = []
    current_entity_type = None  

    def get_entity_type(label_id):
        """Extract entity type from BIO label ID (e.g., 'B-parish' -> 'parish')."""
        if label_id == o_label_id or label_id not in id2label:
             return None
        label = id2label[label_id]
        return label[2:] if label.startswith(('B-', 'I-')) else None

    def merge_entity(stack_bboxes, stack_tokens, entity_type):
        """Merge stacked boxes and tokens into a single entity."""

        valid_bboxes_np = np.array(stack_bboxes)
        merged_bbox = [
            np.min(valid_bboxes_np[:, 0]), # min x1
            np.min(valid_bboxes_np[:, 1]), # min y1
            np.max(valid_bboxes_np[:, 2]), # max x2
            np.max(valid_bboxes_np[:, 3]), # max y2
        ]
        merged_token = " ".join(stack_tokens) 
        return merged_bbox, merged_token, entity_type


    for i, (bbox, prediction, token) in enumerate(zip(bboxes, predictions, tokens)):
        label = id2label.get(prediction)
        if label is None:
             if verbose:
                 print(f"Warning: Invalid prediction ID {prediction} at index {i}. Treating as O.")
             prediction = o_label_id # Treat as O
             label = "O"

        new_entity_type = get_entity_type(prediction)

        # Case 1: Current token is O
        if prediction == o_label_id:
            if bbox_stack: # Finalize the previous entity
                merged_bbox, merged_token, entity_class = merge_entity(
                    bbox_stack, token_stack, current_entity_type
                )
                if merged_bbox:
                    merged_boxes.append(merged_bbox)
                    merged_tokens.append(merged_token)
                    merged_classes.append(entity_class)
                    if verbose: print(f"Finalized entity (due to O): {entity_class} -> '{merged_token}'")
                bbox_stack = []
                token_stack = []
                current_entity_type = None
            # Do nothing else for O

        # Case 2: Current token is B-
        elif label.startswith('B-'):
            if bbox_stack: # Finalize the previous entity first
                merged_bbox, merged_token, entity_class = merge_entity(
                    bbox_stack, token_stack, current_entity_type
                )
                if merged_bbox:
                    merged_boxes.append(merged_bbox)
                    merged_tokens.append(merged_token)
                    merged_classes.append(entity_class)
                    if verbose: print(f"Finalized entity (due to B-): {entity_class} -> '{merged_token}'")

            # Start new entity
            bbox_stack = [bbox]
            token_stack = [str(token)] # Ensure token is string
            current_entity_type = new_entity_type
            if verbose:
                print(f"Starting new entity: {current_entity_type}, token: {token}")

        # Case 3: Current token is I-
        elif label.startswith('I-'):
            # Ensure new_entity_type is valid before proceeding
            if new_entity_type is None:
                 if verbose: print(f"Warning: I- tag {label} for token {token} has no valid entity type. Treating as O.")
                 # Finalize previous entity if any
                 if bbox_stack:
                     merged_bbox, merged_token, entity_class = merge_entity(bbox_stack, token_stack, current_entity_type)
                     if merged_bbox:
                         merged_boxes.append(merged_bbox)
                         merged_tokens.append(merged_token)
                         merged_classes.append(entity_class)
                         if verbose: print(f"Finalized entity (due to invalid I- type): {entity_class} -> '{merged_token}'")
                     bbox_stack = []
                     token_stack = []
                     current_entity_type = None
                 continue # Skip to next token


            if bbox_stack and current_entity_type == new_entity_type:
                # Check for line break or large gap before continuing
                last_bbox = bbox_stack[-1]
                current_bbox = bbox

                # Heuristic thresholds (adjust as needed)
                max_vertical_dist_factor = 0.7 # Allow center diff up to 70% of last box height
                max_horizontal_gap_factor = 2.0 # Allow gap up to 1.5x width of last box (reduced)

                last_height = last_bbox[3] - last_bbox[1]
                last_width = last_bbox[2] - last_bbox[0]
                last_center_y = (last_bbox[1] + last_bbox[3]) / 2
                current_center_y = (current_bbox[1] + current_bbox[3]) / 2

                vertical_dist = abs(current_center_y - last_center_y)
                horizontal_gap = current_bbox[0] - last_bbox[2] # Positive -> gap, Negative -> overlap

                is_break = False
                # 1. Check for significant vertical distance (likely new line)
                # Check if last_height is positive to avoid division by zero or nonsensical checks
                if last_height > 1 and vertical_dist > last_height * max_vertical_dist_factor:
                     is_break = True
                     if verbose: print(f"Break detected (Vertical): Entity {current_entity_type}, Token '{token}', VDist: {vertical_dist:.1f} > Threshold: {last_height * max_vertical_dist_factor:.1f}")
                # 2. Check for large horizontal gap (only if vertically close)
                # Check if last_width is positive
                elif last_width > 1 and horizontal_gap > last_width * max_horizontal_gap_factor and vertical_dist <= last_height * max_vertical_dist_factor:
                     is_break = True
                     if verbose: print(f"Break detected (Horizontal Gap): Entity {current_entity_type}, Token '{token}', HGap: {horizontal_gap:.1f} > Threshold: {last_width * max_horizontal_gap_factor:.1f}")
                # 3. Check for wrap-around (significant move left AND down)
                # Check if last_width is positive
                elif last_width > 1 and current_bbox[0] < (last_bbox[0] - last_width * 0.5) and current_center_y > last_center_y + last_height * 0.1: # Added small vertical threshold
                     is_break = True
                     if verbose: print(f"Break detected (Wrap Around): Entity {current_entity_type}, Token '{token}', X1_curr: {current_bbox[0]:.1f} < Threshold: {last_bbox[0] - last_width * 0.5:.1f}")


                if not is_break:
                    # Continue the current entity
                    bbox_stack.append(current_bbox)
                    token_stack.append(str(token))
                    if verbose:
                        print(f"Continuing entity: {current_entity_type}, token: {token}")
                else:
                    # Finalize the previous entity due to break
                    merged_bbox, merged_token, entity_class = merge_entity(
                        bbox_stack, token_stack, current_entity_type
                    )
                    if merged_bbox:
                        merged_boxes.append(merged_bbox)
                        merged_tokens.append(merged_token)
                        merged_classes.append(entity_class)
                        if verbose: print(f"Finalized entity (due to break): {entity_class} -> '{merged_token}'")

                    # Start new entity with the current I- token (treat as B-)
                    bbox_stack = [current_bbox]
                    token_stack = [str(token)]
                    # current_entity_type remains new_entity_type (which is same as old one here)
                    if verbose:
                        print(f"Starting new entity (from I- after break): {current_entity_type}, token: {token}")

            else:
                # Invalid I-: Mismatched type or I- follows O/Start. Treat as B-.
                if verbose:
                    reason = "follows O/Start/Invalid" if not bbox_stack else f"mismatched type (expected {current_entity_type}, got {new_entity_type})"
                    print(f"Warning: Invalid I- tag {label} for token {token} ({reason}). Treating as B-{new_entity_type}.")

                if bbox_stack: # Finalize the previous entity if it exists
                    merged_bbox, merged_token, entity_class = merge_entity(
                        bbox_stack, token_stack, current_entity_type
                    )
                    if merged_bbox:
                        merged_boxes.append(merged_bbox)
                        merged_tokens.append(merged_token)
                        merged_classes.append(entity_class)
                        if verbose: print(f"Finalized entity (due to invalid I-): {entity_class} -> '{merged_token}'")

                # Start new entity based on this "invalid" I- tag's type
                bbox_stack = [bbox] # Use the original bbox variable here
                token_stack = [str(token)] # Ensure token is string
                current_entity_type = new_entity_type
                if verbose:
                    print(f"Starting new entity (from invalid I-): {current_entity_type}, token: {token}")

    # Final check: Merge any remaining entity after the loop
    if bbox_stack:
        merged_bbox, merged_token, entity_class = merge_entity(
            bbox_stack, token_stack, current_entity_type
        )
        if merged_bbox:
            merged_boxes.append(merged_bbox)
            merged_tokens.append(merged_token)
            merged_classes.append(entity_class)
            if verbose: print(f"Finalized entity (end of list): {entity_class} -> '{merged_token}'")

    # Final validation for visualization function compatibility
    if not (len(merged_boxes) == len(merged_tokens) == len(merged_classes)):
         print("Error: Length mismatch after merging! Check verbose logs.")
         print(f"Boxes: {len(merged_boxes)}, Tokens: {len(merged_tokens)}, Classes: {len(merged_classes)}")
         # Depending on severity, you might want to return empty lists or raise an error
         # return [], [], []


    return merged_boxes, merged_tokens, merged_classes
