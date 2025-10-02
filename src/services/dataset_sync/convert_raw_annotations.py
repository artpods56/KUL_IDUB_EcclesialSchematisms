# Jupyter / one-cell script ----------------------------------------------------
"""
Convert Label Studio object-detection boxes + PyTesseract OCR
into BIO-tagged token data for LayoutLMv3.

Usage inside the notebook:
    !python convert.py \
        --image_dir  ./images \
        --label_dir  ./label_studio_json \
        --out_jsonl  ./layoutlm_data.jsonl

Or just tweak the paths below and run this cell.
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from shapely.geometry import box
import geopandas as gpd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from huggingface_hub import HfApi, Repository
import subprocess
from datetime import datetime

from config import setup_logging, load_config_from_env


def run_git_command(
    command: List[str], working_dir: Path, raise_on_error: bool = True
) -> Optional[subprocess.CompletedProcess]:
    """Helper function to run a git command."""
    logging.debug(f"Running git command: {' '.join(command)} in {working_dir}")
    try:
        process = subprocess.run(
            command,
            cwd=working_dir,
            check=False,  # We'll check manually to provide better logging
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            error_message = (
                f"Git command failed: {' '.join(command)}\n"
                f"Return Code: {process.returncode}\n"
                f"Stdout: {process.stdout.strip()}\n"
                f"Stderr: {process.stderr.strip()}"
            )
            logging.error(error_message)
            if raise_on_error:
                raise subprocess.CalledProcessError(
                    process.returncode, command, process.stdout, process.stderr
                )
            return None
        logging.debug(f"Git command successful. Stdout: {process.stdout.strip()}")
        return process
    except FileNotFoundError:
        logging.error("Git command not found. Ensure git is installed and in PATH.")
        if raise_on_error:
            raise
        return None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred while running git command {' '.join(command)}: {e}"
        )
        if raise_on_error:
            raise
        return None


def is_repo_dirty(repo_dir: Path) -> bool:
    """Checks if the git repository has uncommitted changes."""
    process = run_git_command(
        ["git", "status", "--porcelain"], repo_dir, raise_on_error=False
    )
    if process and process.stdout.strip():
        logging.debug(f"Repository is dirty. Status:\n{process.stdout.strip()}")
        return True
    logging.debug("Repository is clean.")
    return False


def push_to_hf_repo(config, commit_message_metadata=""):
    """Push the generated train.jsonl to HuggingFace data repository using direct git commands."""

    commit_message = f"data update: {commit_message_metadata} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    try:
        hf_repo_dir = Path(config["HF_REPO_DIR"])
        # config['OUT_JSONL'] is a Path object representing the relative path like 'labels/train.jsonl'
        relative_output_path = config["OUT_JSONL"]
        hf_token = config.get("HF_TOKEN")  # Get token, might be None

        if not hf_repo_dir.is_dir() or not (hf_repo_dir / ".git").is_dir():
            logging.error(
                f"HF_REPO_DIR {hf_repo_dir} is not a valid git repository. Skipping push."
            )
            # This case should ideally be handled by setup_data.sh ensuring the repo is cloned.
            return

        # Check for and remove stale git lock file
        lock_file = hf_repo_dir / ".git" / "index.lock"
        if lock_file.exists():
            logging.warning(f"Stale git lock file found at {lock_file}. Removing it.")
            try:
                lock_file.unlink()
            except OSError as e:
                logging.error(
                    f"Failed to remove lock file {lock_file}: {e}. Git operations might fail."
                )
                # Not raising here, to allow git commands to try and potentially provide more specific errors

        # 1. Git Pull
        logging.info(
            f"Pulling latest changes from remote repository in {hf_repo_dir}..."
        )
        # Assuming the remote 'origin' is already configured correctly (e.g., by setup_data.sh with token)
        run_git_command(["git", "pull"], hf_repo_dir)

        logging.info(f"Contents of {hf_repo_dir} after git pull:")
        try:
            for item in hf_repo_dir.iterdir():
                logging.info(f"  - {item.name}")
            status_result = run_git_command(
                ["git", "status", "-s"], hf_repo_dir, raise_on_error=False
            )
            if status_result:
                logging.info(f"Git status after pull:\n{status_result.stdout}")
        except Exception as e_log:
            logging.error(f"Error listing directory contents: {e_log}")

        # 2. Git Add
        # The file to add is relative_output_path (e.g., "labels/train.jsonl")
        # Ensure the file actually exists before trying to add it
        full_output_file_path = hf_repo_dir / relative_output_path
        if not full_output_file_path.exists():
            logging.error(
                f"Output file {full_output_file_path} does not exist. Cannot add to git."
            )
            return  # Or raise an error

        logging.info(f"Adding {relative_output_path} to git...")
        run_git_command(["git", "add", str(relative_output_path)], hf_repo_dir)

        # 3. Git Commit (only if there are changes)
        if is_repo_dirty(hf_repo_dir):
            logging.info(f"Committing changes with message: {commit_message}")
            run_git_command(["git", "commit", "-m", commit_message], hf_repo_dir)

            # 4. Git Push
            logging.info("Pushing changes to remote repository...")
            # If setup_data.sh cloned with token in URL, this should work.
            # Otherwise, git needs to be configured with credentials (e.g., via HfFolder.save_token or credential helper)
            # For explicit token usage with push (if needed and remote not pre-configured with token):
            # push_command = ["git", "push"]
            # if hf_token:
            #    # This method of embedding token in URL for push is one way,
            #    # but assumes 'origin' is the remote description and needs careful handling of the URL.
            #    # A more robust way if 'origin' isn't set up with token is to use http.extraheader
            #    # push_command = ["git", "-c", f"http.extraheader=AUTHORIZATION: Bearer {hf_token}", "push"]
            #    # However, if clone used token in URL, 'git push' should suffice.
            run_git_command(["git", "push"], hf_repo_dir)
            logging.info(
                f"Successfully pushed changes to HF repository: {config['HF_REPO_URL']}"
            )
        else:
            logging.info("No changes to commit or push.")

    except subprocess.CalledProcessError as e:
        # Error is already logged by run_git_command
        logging.error(f"A git command failed during push_to_hf_repo. See logs above.")
        raise  # Re-raise to indicate failure to the caller
    except Exception as e:
        logging.error(
            f"An unexpected error occurred in push_to_hf_repo: {e}", exc_info=True
        )
        raise


def s3_path_to_identifier(s3_path: Path) -> str:
    # s3://schematyzmy/tarnow_1870/0013.jpg
    # into tarnow_1870_0013
    return "_".join(s3_path.parts[-2:]).replace(".jpg", "").replace(".png", "")


def ls_to_xyxy(ls_boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Convert Label Studio % boxes -> absolute xyxy pixels.
    ls_boxes: shape (N, 4) with columns [x%, y%, w%, h%]
    """
    b = ls_boxes.astype(float).copy()
    b[:, [0, 2]] *= w / 100.0  # x%, w% -> pixels
    b[:, [1, 3]] *= h / 100.0  # y%, h% -> pixels
    b[:, 2] += b[:, 0]  # w -> xmax
    b[:, 3] += b[:, 1]  # h -> ymax
    return b


def check_if_ocr_exists(image_path: Path, ocr_dir: Path) -> bool:
    """
    Check if OCR data exists for the given image.
    Returns True if OCR file exists, False otherwise.
    """
    ocr_file = ocr_dir / (image_path.stem + ".json")
    return ocr_file.exists()


def run_ocr(
    image_pil: Image.Image, image_path: Path, ocr_dir: Path
) -> Tuple[List[str], List[List[int]], List[float]]:
    """
    Run Tesseract, then tokenize words (sub-word duplication keeps coordinates aligned).
    Returns tokens, token_boxes (xyxy), confidences
    """

    if check_if_ocr_exists(image_path, ocr_dir):
        with open(ocr_dir / (image_path.stem + ".json"), "r") as f:
            ocr_data = json.load(f)

            tokens = ocr_data["tokens"]
            boxes = ocr_data["boxes"]
            confs = ocr_data["confs"]
            return tokens, boxes, confs
    else:
        ocr = pytesseract.image_to_data(
            np.array(image_pil.convert("L")),
            output_type=pytesseract.Output.DICT,
            lang="lat+pol+rus",
        )

        tokens, boxes, confs = [], [], []

        for i in range(len(ocr["text"])):
            if ocr["level"][i] != 5:  # keep only word level
                continue
            word = ocr["text"][i].strip()
            if not word:  # drop blanks early
                continue

            xmin = ocr["left"][i]
            ymin = ocr["top"][i]
            xmax = xmin + ocr["width"][i]
            ymax = ymin + ocr["height"][i]
            word_box = [xmin, ymin, xmax, ymax]
            word_conf = float(ocr["conf"][i])

            # sub-tokenise but keep same bbox for every sub-token

            tokens.append(word)
            boxes.append(word_box)
            confs.append(word_conf)

        # Save OCR data for later use
        ocr_data = {"tokens": tokens, "boxes": boxes, "confs": confs}
        ocr_file = ocr_dir / (image_path.stem + ".json")
        ocr_file.parent.mkdir(parents=True, exist_ok=True)
        with open(ocr_file, "w") as f:
            json.dump(ocr_data, f, ensure_ascii=False, indent=4)

        return tokens, boxes, confs


def normalize_bbox(b: List[int], w: int, h: int) -> List[int]:
    """
    LayoutLM expects integer coords in [0,1000].
    """
    return [
        int(1000 * b[0] / w),
        int(1000 * b[1] / h),
        int(1000 * b[2] / w),
        int(1000 * b[3] / h),
    ]


def assign_bio(
    token_boxes: List[List[int]],
    ent_boxes: np.ndarray,
    ent_labels: List[str],
    iou_thr: float = 0.3,
    overlap_thr: float = 0.7,
) -> List[str]:
    """
    Map every token bbox to the best-overlapping entity.
    Returns BIO tag list same length as token_boxes.
    """
    # Build GeoDataFrames once for vectorised spatial join
    tok_gdf = gpd.GeoDataFrame(
        {"idx": range(len(token_boxes)), "geometry": [box(*b) for b in token_boxes]},
        crs=None,
    )
    ent_gdf = gpd.GeoDataFrame(
        {
            "eidx": range(len(ent_boxes)),
            "label": ent_labels,
            "geometry": [box(*b) for b in ent_boxes],
        },
        crs=None,
    )

    cand = gpd.sjoin(tok_gdf, ent_gdf, how="inner", predicate="intersects")

    # Only proceed if we have candidates
    if len(cand) == 0:
        return ["O"] * len(token_boxes)

    # Measure actual overlap
    def _metrics(r):
        t = r.geometry
        e = ent_gdf.loc[r.eidx].geometry
        inter = t.intersection(e).area
        union = t.union(e).area
        t_area = t.area
        centroid_in = t.centroid.within(e)
        return pd.Series(
            {
                "iou": inter / union if union else 0.0,
                "overlap": inter / t_area if t_area else 0.0,
                "centroid_in": centroid_in,
            }
        )

    cand[["iou", "overlap", "centroid_in"]] = cand.apply(_metrics, axis=1)

    # Keep only candidates clearing either threshold
    # cand = cand[(cand.iou >= iou_thr) | (cand.overlap >= overlap_thr) | (cand.centroid_in)]
    cand = cand[cand.centroid_in]
    # Choose best match per token
    best = cand.sort_values(["idx", "iou", "overlap"], ascending=False).drop_duplicates(
        "idx"
    )

    bio = ["O"] * len(token_boxes)

    # Group by entity to ensure proper BIO tagging
    if len(best) > 0:
        grouped = best.groupby("eidx")

        for _, grp in grouped:
            lbl = grp.iloc[0]["label"]
            idxs = sorted(grp["idx"].tolist())  # Sort to maintain order
            if idxs:  # Make sure we have indices
                bio[idxs[0]] = f"B-{lbl}"
                for i in idxs[1:]:
                    bio[i] = f"I-{lbl}"

    return bio


def process_one(annotation_path: Path, tokenizer, image_dir, ocr_cache_dir: Path):
    # ---------- image ---------- #
    with open(annotation_path, "r") as f:
        ls = json.load(f)

    local_identifier = s3_path_to_identifier(Path(ls["task"]["data"]["image"]))

    image_path = image_dir / (local_identifier + ".jpg")
    json_path = annotation_path  # Use the actual annotation file path

    with Image.open(image_path) as image_pil:
        W, H = image_pil.size
        words, t_boxes, confs = run_ocr(image_pil, image_path, ocr_cache_dir)

    # ---------- Label Studio ---------- #

    ls_boxes_pct, ls_labels = [], []
    for ann in ls["result"]:
        v = ann["value"]
        ls_boxes_pct.append([v["x"], v["y"], v["width"], v["height"]])
        ls_labels.append(v["rectanglelabels"][0])

    if not ls_boxes_pct:
        ent_boxes = np.empty((0, 4))
    else:
        ent_boxes = ls_to_xyxy(np.asarray(ls_boxes_pct), W, H)

    # ---------- BIO assignment ---------- #
    labels = assign_bio(t_boxes, ent_boxes, ls_labels)

    # ---------- HuggingFace sample ---------- #
    objects = []
    for ls_bbox, ls_label in zip(ls_boxes_pct, ls_labels):
        objects.append(
            {
                "class": ls_label,
                "bbox": {
                    "xmin": int(ls_bbox[0]),
                    "ymin": int(ls_bbox[1]),
                    "xmax": int((ls_bbox[0] + ls_bbox[2])),
                    "ymax": int((ls_bbox[1] + ls_bbox[3])),
                },
            }
        )

    return {
        "image_path": str(image_path).split("/")[-1],
        "width": W,
        "height": H,
        "words": words,
        "bboxes": [normalize_bbox(b, W, H) for b in t_boxes],
        "labels": labels,
        "conf": confs,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw Label Studio annotations to LayoutLMv3 file_format."
    )
    parser.add_argument(
        "--env_file",
        type=str,
        default=".env",
        help="Path to the .env file for configuration.",
    )
    parser.add_argument(
        "--commit_message_metadata",
        type=str,
        default=None,
        help="Dataset diff for commit message.",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    config = load_config_from_env(cli_args.env_file)

    setup_logging(config["LOG_LEVEL"])

    ls_annotations_dir = Path(config["LS_ANNOTATIONS_DIR"])
    logging.info(
        f"Converting Label Studio annotations from {ls_annotations_dir} to LayoutLMv3 file_format."
    )

    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-large")

    samples = []
    anotations_files = sorted(
        [f for f in os.listdir(ls_annotations_dir) if f.endswith(".json")]
    )

    for annotation_file in tqdm(anotations_files, desc="Processing annotations"):
        annotation_path = ls_annotations_dir / annotation_file
        try:
            processed_sample = process_one(
                annotation_path, tokenizer, config["IMAGE_DIR"], config["OCR_CACHE_DIR"]
            )
            samples.append(processed_sample)
        except Exception as e:
            logging.error(f"Failed to process {annotation_file}: {e}")

    with (config["HF_REPO_DIR"] / config["OUT_JSONL"]).open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logging.info(f"✓ Wrote {len(samples)} samples → {config['OUT_JSONL']}")
    logging.info("Pushing to HuggingFace repository...")
    push_to_hf_repo(config, cli_args.commit_message_metadata)


if __name__ == "__main__":
    main()

# command line to run this script:
# python convert_raw_annotations.py --env_file .env
