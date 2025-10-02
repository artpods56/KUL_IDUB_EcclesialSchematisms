from typing import Callable
from thefuzz import fuzz, process

from core.schemas.data.metrics import Metrics, PageDataMetrics
from core.schemas.data.schematism import SchematismPage, SchematismEntry
from structlog import get_logger
import unicodedata
import re

logger = get_logger(__name__)


def normalize_text(txt: str | None) -> str:
    if not txt:
        return ""
    return (
        unicodedata.normalize("NFKD", txt)
        .encode("ascii", "ignore")
        .decode()
        .strip(",.; ")
        .lower()
    )


def fallback_scorer(a: str, b: str) -> int:
    """
    Fuzzy scorer with fallback: tries original token_set_ratio,
    and also tries after stripping leading 'number. ' patterns.
    Returns the maximum score.
    """
    score1 = fuzz.token_set_ratio(a, b)
    
    # Strip leading number. from both strings
    stripped_a = re.sub(r'^\d+\.\s*', '', a)
    stripped_b = re.sub(r'^\d+\.\s*', '', b)
    
    score2 = fuzz.token_set_ratio(stripped_a, stripped_b)
    
    return max(score1, score2)


def normalize_text(txt: str | None) -> str:
    if not txt:
        return ""
    return (
        unicodedata.normalize("NFKD", txt)
        .encode("ascii", "ignore")
        .decode()
        .strip(",.; ")
        .lower()
    )


def normalize_page_data(page_data: SchematismPage) -> SchematismPage:
    return SchematismPage(
        page_number=page_data.page_number,
        entries=[
            SchematismEntry(
                parish=normalize_text(entry.parish),
                deanery=normalize_text(entry.deanery),
                dedication=normalize_text(entry.dedication),
                building_material=normalize_text(entry.building_material),
            )
            for entry in page_data.entries
        ]
    )

def align_page_data(
    predictions_page_data: SchematismPage,
    ground_truth_page_data: SchematismPage,
    fuzzy_threshold: int = 80,
    scorer: Callable[..., int] = fallback_scorer,
) -> list[tuple[SchematismEntry | None, SchematismEntry | None]]:
    """
    Aligns entries from prediction and ground truth SchematismPage based on fuzzy matching of the 'parish' field.

    Returns a list of tuples, where each tuple contains a matched pair of (prediction, ground_truth) entries.
    Unmatched entries are paired with None.
    """
    pred_entries = predictions_page_data.entries
    gt_entries = ground_truth_page_data.entries

    pred_parishes = {i: e.parish for i, e in enumerate(pred_entries)}
    gt_parishes = {i: e.parish for i, e in enumerate(gt_entries)}

    matches: list[tuple[SchematismEntry, SchematismEntry]] = []
    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()

    for i, gt_parish in gt_parishes.items():
        if not gt_parish:
            continue

        unmatched_preds = {
            p_val: p_idx for p_idx, p_val in pred_parishes.items() if p_idx not in matched_pred_indices and p_val
        }

        if not unmatched_preds:
            break

        result = process.extractOne(gt_parish, unmatched_preds.keys(), scorer=scorer)
        if result:
            best_match_parish, score = result
            if score >= fuzzy_threshold:
                pred_idx = unmatched_preds[best_match_parish]

                matches.append((pred_entries[pred_idx], gt_entries[i]))
                matched_pred_indices.add(pred_idx)
                matched_gt_indices.add(i)

    aligned_pairs: list[tuple[SchematismEntry | None, SchematismEntry | None]] = []
    aligned_pairs.extend(matches)

    for i, pred_entry in enumerate(pred_entries):
        if i not in matched_pred_indices:
            aligned_pairs.append((pred_entry, None))

    for i, gt_entry in enumerate(gt_entries):
        if i not in matched_gt_indices:
            aligned_pairs.append((None, gt_entry))

    return aligned_pairs


def evaluate_json_response(predictions_data: SchematismPage,
                           ground_truth_data: SchematismPage,
                           fuzzy_threshold: int = 80,
                           scorer: Callable[...,int] = fallback_scorer,
                           ) -> PageDataMetrics:

    entry_fields = list(SchematismEntry.model_fields.keys())
    metrics = {
        field: Metrics() for field in ["page_number", *entry_fields]
    }

    normalized_predictions = normalize_page_data(predictions_data)
    normalized_ground_truth = normalize_page_data(ground_truth_data)

    # Handle special case: if ground truth has no page_number (indicating unannotated empty page)
    # and no entries, then any prediction should get F1=1.0 (perfect score)
    gt_pn = normalized_ground_truth.page_number
    pred_pn = normalized_predictions.page_number
    gt_entries = normalized_ground_truth.entries
    pred_entries = normalized_predictions.entries
    
    # Handle page_number field
    if gt_pn and pred_pn:
        if scorer(gt_pn, pred_pn) >= fuzzy_threshold:
            metrics["page_number"].update(tp=1)
        else:
            metrics["page_number"].update(fp=1, fn=1)
    elif gt_pn and not pred_pn:
        metrics["page_number"].update(fn=1)
    elif not gt_pn and pred_pn:
        # Special case: when ground truth has no page_number (unannotated page), 
        # give perfect score for page_number field regardless of prediction
        metrics["page_number"].update(tp=1)
    elif not gt_pn and not pred_pn:
        # Both empty - perfect score
        metrics["page_number"].update(tp=1)

    # Handle entries alignment and scoring
    # Special case: both predictions and ground truth have no entries -> perfect F1=1.0
    if not pred_entries and not gt_entries:
        for field in entry_fields:
            metrics[field].update(tp=1)
    else:
        aligned_pairs = align_page_data(
            normalized_predictions, normalized_ground_truth, fuzzy_threshold, scorer
        )

        for pred_entry, gt_entry in aligned_pairs:

            for field in entry_fields:
                pred_val = getattr(pred_entry, field, None) if pred_entry else None
                gt_val = getattr(gt_entry, field, None) if gt_entry else None

                # logger.info(f"{field} - pred_val: {pred_val}, gt_val: {gt_val}")
                if gt_val and pred_val:
                    if scorer(pred_val, gt_val) >= fuzzy_threshold:
                        metrics[field].update(tp=1)
                    else:
                        metrics[field].update(fp=1, fn=1)
                elif gt_val and not pred_val:
                    metrics[field].update(fn=1)
                elif not gt_val and pred_val:
                    metrics[field].update(fp=1)

    # logger.info(f"Metrics: {metrics}")
    return PageDataMetrics(**metrics)
