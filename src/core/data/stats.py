from typing import Dict
from thefuzz import fuzz, process


from core.utils.logging import setup_logging
from logging import getLogger

setup_logging()


logger = getLogger(__name__)



PAGE_ONLY = {"B-page_number", "I-page_number", "O"}

def compute_dataset_stats(dataset) -> Dict:
    counter_template = {
        "positive": 0,
        "negative": 0,
        "page_number_only": 0,
    }

    schematisms = {}

    for ex in dataset:
        full_filename = ex["image"]
        splits = full_filename.split("_")
        filename = splits.pop()
        schematism = "_".join(splits)

        if schematism not in schematisms:
            schematisms[schematism] = counter_template.copy()

        lbl_set = set(ex["labels"]) 

        if lbl_set - {"O"}:
            schematisms[schematism]["positive"] += 1
        else:
            schematisms[schematism]["negative"] += 1

        if lbl_set <= PAGE_ONLY:
            schematisms[schematism]["page_number_only"] += 1

    total = len(dataset)
    stats = {
        "total_examples": total,
        "positive": sum(s["positive"] for s in schematisms.values()),
        "negative": sum(s["negative"] for s in schematisms.values()),
        "neg_to_pos_ratio": (
            sum(s["negative"] for s in schematisms.values())
            / sum(s["positive"] for s in schematisms.values())
            if sum(s["positive"] for s in schematisms.values())
            else None
        ),
        "page_number_only": sum(s["page_number_only"] for s in schematisms.values()),
        "page_number_only_ratio": (
            sum(s["page_number_only"] for s in schematisms.values()) / total
            if total
            else None
        ),
        "schematisms_stats": schematisms,
    }

    return stats

def evaluate_entries(pred_entries, gt_entries, fuzzy_threshold=80, scorer=fuzz.partial_ratio):
    """
    Evaluates entries using fuzzy matching and returns TP, FP, FN counts.
    """

    tp, fp, fn = 0,0,0
    
    matched = []
    
    standarized_predicted_entries = [normalize_entry(e) for e in pred_entries]
    standarized_gt_entries = [normalize_entry(e) for e in gt_entries]
    
    predicted_parishes = [e['parish'] for e in standarized_predicted_entries]
    unmatched_pred = list(predicted_parishes)

    
    logger.info(f"[STATUS] real parishes {[e['parish'] for e in standarized_gt_entries]}")
    for ground_entry in standarized_gt_entries:
        ground_parish = ground_entry['parish']

        logger.info(f"[EVALUATING] searching for: {ground_parish}")
        logger.info(f"[STATUS] available parishes: {unmatched_pred}")
        match = process.extractOne(ground_parish, unmatched_pred, scorer=scorer, score_cutoff=fuzzy_threshold)
        
        if not match:
            logger.info(f"[NO MATCH] No match found for {ground_parish}")
            fn += 1
            continue
        else:
            tp += 1
            best_match = match[0]
            score = match[1]

        unmatched_pred.remove(best_match)  # Remove matched parish from unmatched list  

        logger.info(f"[MATCH] {ground_parish} -> {best_match} with score {score}")        
        # unmatched_gt.remove(gt_match_entry)
        # gt_parishes.remove(best_match)
        
        # if score >= fuzzy_threshold:
        #     logger.info(f"[MATCH] found: {ground_parish} -> {best_match} with score {score}")
        #     logger.info(f"[STATUS] avaiable parishes: {[e['parish'] for e in unmatched_pred]}")
            
        #     gt_match_entry = next((e for e in unmatched_pred if e['parish'] == best_match), None)
            
        #     if gt_match_entry is None:  # Should not happen if logic is correct, but safe
        #         fp += 1
        #         continue


        #     is_correct = True
        #     for field in ['dedication', 'building_material']:
        #         pred_val = ground_entry.get(field, '')
        #         gt_val = gt_match_entry.get(field, '')

        #         if fuzz.partial_ratio(pred_val, gt_val) < 60:  # Stricter for content
        #             is_correct = False
        #             logger.info(f"[BAD] field mismatch: {field} pred='{pred_val}' gt='{gt_val}'")
        #             break
            
        #     if is_correct:
        #         logger.info(f"[GOOD] correct: {pred_parish} -> {best_match} with score {score}")
        #         tp += 1
        #     else:
        #         logger.info(f"[BAD] incorrect: {pred_parish} -> {best_match} with score {score}")
        #         fp += 1  # Matched the parish, but content was wrong
            
        #     # Remove the matched entry and its parish name from our lists to avoid re-matching

        # else:
        #     # The best match was not good enough, this is a hallucinated entry
        #     logger.info(f"[BAD] no match found for {pred_parish}, score {score}")
        #     fp += 1

    # Any remaining entries in unmatched_gt are ones the model missed

    fp = len(unmatched_pred)  # All unmatched predicted entries are false positives
    return tp, fp, fn

from rapidfuzz import process, fuzz
import unicodedata


# -------------------------  normalizacja  -------------------------
def normalize_text(txt: str | None) -> str:
    if txt is None:
        return ""
    txt = txt.strip(",.; ")              # trailing comma/kropka
    txt = unicodedata.normalize("NFKD", txt)
    txt = txt.encode("ascii", "ignore").decode()
    return txt.lower()

def normalize_entry(entry: dict) -> dict:
    return {k: normalize_text(v) if isinstance(v, str) else v
            for k, v in entry.items()}

# -------------------------  metryki pomocnicze  --------------------
def _init_metrics():
    """Create a fresh metric dictionary initialising all counters and rates."""
    return {"TP": 0, "FP": 0, "FN": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

def _update(metrics: dict, field: str, tp=0, fp=0, fn=0):
    m = metrics[field]
    m["TP"] += tp
    m["FP"] += fp
    m["FN"] += fn

def _precision_recall(m):
    tp, fp, fn = m["TP"], m["FP"], m["FN"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec

# helper
def _f1(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def _accuracy(m):
    total = m["TP"] + m["FP"] + m["FN"]
    return m["TP"] / total if total else 0.0

# -------------------------  główna funkcja  -----------------------
def evaluate_json_response(pred_json: dict,
                           gt_json: dict,
                           fuzzy_threshold: int = 80,
                           scorer = fuzz.token_set_ratio,
                           entry_fields = ("parish", "dedication",
                                           "building_material")) -> dict:
    """
    Zwraca słownik: {pole: {"TP":..., "FP":..., "FN":..., "precision":..., "recall":...}}
    """
    metrics = {field: _init_metrics() for field in
               ["page_number", "deanery", *entry_fields]}

    # --- pola globalne ---
    for field in ("page_number", "deanery"):
        gt_val  = normalize_text(gt_json.get(field))
        pred_val = normalize_text(pred_json.get(field))
        if not gt_val:                         # ground truth puste
            if pred_val: _update(metrics, field, fp=1)
        elif not pred_val:                     # brak predykcji
            _update(metrics, field, fn=1)
        elif scorer(gt_val, pred_val) >= fuzzy_threshold:
            _update(metrics, field, tp=1)
        else:                                  # błędna wartość
            _update(metrics, field, fp=1, fn=1)

    # --- wpisy (entries) ---
    gt_entries   = gt_json["entries"]
    pred_entries = pred_json["entries"]

    # lista nieprzypisanych predicted parishes
    available_parishes = [e["parish"] for e in pred_entries]

    for gt_entry in gt_entries:
        gt_parish = gt_entry["parish"]
        match_result = process.extractOne(
            gt_parish, available_parishes,
            scorer=scorer, score_cutoff=fuzzy_threshold
        )
        if not match_result:
            # brak parafii => FN dla każdego pola z entry_fields
            for fld in entry_fields:
                _update(metrics, fld, fn=1)
            continue

        best_match_parish, score, index = match_result
        # zdejmujemy z puli
        best_pred_entry = next(e for e in pred_entries
                               if e["parish"] == best_match_parish)
        available_parishes.remove(best_match_parish)

        # parish trafiony:
        _update(metrics, "parish", tp=1)

        # pozostałe pola
        for fld in entry_fields[1:]:           # bez 'parish'
            gt_val   = gt_entry.get(fld, "")
            pred_val = best_pred_entry.get(fld, "")
            if not gt_val:
                if pred_val: _update(metrics, fld, fp=1)
            elif not pred_val:
                _update(metrics, fld, fn=1)
            elif scorer(gt_val, pred_val) >= fuzzy_threshold:
                _update(metrics, fld, tp=1)
            else:
                _update(metrics, fld, fp=1, fn=1)

    # pozostałe predicted, które nie dostały matcha → FP dla każdego pola
    for _ in available_parishes:
        for fld in entry_fields:
            _update(metrics, fld, fp=1)

    # --- uzupełnij prec/rec ---
    for fld in metrics:
        prec, rec = _precision_recall(metrics[fld])
        metrics[fld]["precision"] = round(prec, 3)
        metrics[fld]["recall"]    = round(rec, 3)
        f1_score = _f1(prec, rec)
        acc = _accuracy(metrics[fld])
        metrics[fld]["f1"] = round(f1_score, 3)
        metrics[fld]["accuracy"] = round(acc, 3)

    return metrics
