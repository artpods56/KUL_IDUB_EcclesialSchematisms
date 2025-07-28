import os
import json
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import structlog

from core.config.constants import ConfigType, ModelsConfigSubtype
from core.config.helpers import with_configs
from omegaconf import DictConfig

from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model
from core.models.ocr.model import OcrModel
from core.data.translation_parser import Parser
from core.utils.shared import TMP_DIR
from core.utils.logging import setup_logging

import core.schemas.configs

# wandb integration
import wandb
from core.utils.wandb_eval import create_eval_table, add_eval_row

setup_logging()
logger = structlog.get_logger(__name__)

from dotenv import load_dotenv
envs = load_dotenv()
if envs:
    logger.info("Loaded environment variables.")
else:
    logger.warning("No environment variables loaded.")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")


def sort_entries_by_parish(prediction_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sort entries in a prediction dictionary by the 'parish' field.
    Case-insensitive sorting with empty strings as fallback.
    """
    if not isinstance(prediction_dict, dict):
        return prediction_dict
    
    # Create a copy to avoid modifying the original
    sorted_dict = prediction_dict.copy()
    
    # Sort entries if they exist
    if "entries" in sorted_dict and isinstance(sorted_dict["entries"], list):
        sorted_dict["entries"] = sorted(
            sorted_dict["entries"], 
            key=lambda e: e.get("parish", "").lower()
        )
    
    return sorted_dict


@with_configs(
    llm_model_config=("llm_model_config", ConfigType.MODELS, ModelsConfigSubtype.LLM),
    lmv3_model_config=("lmv3_model_config", ConfigType.MODELS, ModelsConfigSubtype.LMV3)
)
def main(llm_model_config: DictConfig, lmv3_model_config: DictConfig):
    """
    Inference script that processes images from input directory and logs predictions to wandb.
    """
    
    # Initialize models
    llm_model = LLMModel(llm_model_config, test_connection=False)
    lmv3_model = LMv3Model(lmv3_model_config)
    ocr_model = OcrModel()
    parser = Parser()
    
    # Initialize wandb
    run = wandb.init(
        project="ai-osrodek", 
        name="inference_run", 
        mode="online", 
        dir=TMP_DIR
    )
    
    # Get input directory
    input_dir = TMP_DIR / "input"
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} image files to process")

    # ------------------------------------------------------------------
    # Prepare data collection for CSV/XLS export
    # ------------------------------------------------------------------
    rows: list[Dict[str, Any]] = []  # main table rows
    debug_rows: list[Dict[str, Any]] = []  # stacked rows with 'source' column for debugging
    
    # Create wandb table for logging predictions
    inference_table = create_eval_table(fields=[])
    
    for image_file in image_files:
        try:
            logger.info(f"Processing {image_file.name}")
            
            # Load image
            image = Image.open(image_file).convert('RGB')
            
            # Prepare metadata
            metadata = {
                "filename": image_file.name,
                "file_path": str(image_file),
            }
            
            # Run inference
            lmv3_prediction = lmv3_model.predict(image, **metadata)
            ocr_text = ocr_model.predict(image, text_only=True, **metadata)
            llm_prediction = llm_model.predict(
                image=image, 
                hints=lmv3_prediction, 
                text=ocr_text, 
                **metadata
            )
            parsed_llm_prediction = parser.parse_page(llm_prediction)
            
            # Sort entries by parish field
            lmv3_prediction_sorted = sort_entries_by_parish(lmv3_prediction)
            llm_prediction_sorted = sort_entries_by_parish(llm_prediction)
            parsed_llm_prediction_sorted = sort_entries_by_parish(parsed_llm_prediction)

            # ----------------------------------------------------------
            # Collect rows for CSV/XLS export
            # ----------------------------------------------------------
            page_level_fields = {
                "filename": image_file.name,
                "page_number": parsed_llm_prediction_sorted.get("page_number"),
                "deanery": parsed_llm_prediction_sorted.get("deanery"),
            }

            for idx, entry in enumerate(parsed_llm_prediction_sorted.get("entries", []), start=1):
                row = {
                    **page_level_fields,
                    "id": f"{image_file.name}_{idx}",
                    "parish": entry.get("parish"),
                    "dedication": entry.get("dedication"),
                    "building_material": entry.get("building_material"),
                }
                rows.append(row)

            # ----------------------------------------------------------
            # Collect DEBUG rows (interleaved per entry: LMv3 → raw LLM → parsed LLM)
            # ----------------------------------------------------------
            lmv3_entries   = lmv3_prediction_sorted.get("entries", [])
            raw_entries    = llm_prediction_sorted.get("entries", [])
            parsed_entries = parsed_llm_prediction_sorted.get("entries", [])

            from itertools import zip_longest

            for entry_idx, (e_lmv3, e_raw, e_parsed) in enumerate(zip_longest(lmv3_entries, raw_entries, parsed_entries), start=1):
                base_id = f"{image_file.name}_{entry_idx}"
                if e_lmv3 is not None:
                    debug_rows.append({
                        **page_level_fields,
                        "id": f"{base_id}_lmv3",
                        "parish": e_lmv3.get("parish"),
                        "dedication": e_lmv3.get("dedication"),
                        "building_material": e_lmv3.get("building_material"),
                        "source": "lmv3",
                    })
                if e_raw is not None:
                    debug_rows.append({
                        **page_level_fields,
                        "id": f"{base_id}_raw_llm",
                        "parish": e_raw.get("parish"),
                        "dedication": e_raw.get("dedication"),
                        "building_material": e_raw.get("building_material"),
                        "source": "raw_llm",
                    })
                if e_parsed is not None:
                    debug_rows.append({
                        **page_level_fields,
                        "id": f"{base_id}_parsed_llm",
                        "parish": e_parsed.get("parish"),
                        "dedication": e_parsed.get("dedication"),
                        "building_material": e_parsed.get("building_material"),
                        "source": "parsed_llm",
                    })

            # Log to wandb table
            add_eval_row(
                table=inference_table,
                sample_id=image_file.name,
                pil_image=image,
                page_info_json={},
                lmv3_response=lmv3_prediction_sorted,
                raw_llm_response=llm_prediction_sorted,
                parsed_llm_response=parsed_llm_prediction_sorted,
                metrics={},
                fields=[]
            )
            
            # Log individual predictions as scalars (optional)
            run.log({
                f"predictions/{image_file.stem}/lmv3_prediction": lmv3_prediction_sorted,
                f"predictions/{image_file.stem}/llm_prediction": llm_prediction_sorted,
                f"predictions/{image_file.stem}/parsed_prediction": parsed_llm_prediction_sorted,
            })
            
            logger.info(f"Successfully processed {image_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing {image_file.name}: {str(e)}")
            continue
    
    # Log the complete table
    run.log({"inference_results": inference_table})

    # ------------------------------------------------------------------
    # Save collected rows to CSV and XLSX
    # ------------------------------------------------------------------
    if rows:
        import pandas as pd
        from datetime import datetime
        from pathlib import Path

        output_dir = TMP_DIR / "inference_tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"inference_results_{timestamp}.csv"
        xlsx_path = output_dir / f"inference_results_{timestamp}.xlsx"

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        try:
            # Pandas uses openpyxl by default for xlsx writing in recent versions.
            df.to_excel(xlsx_path, index=False)
            logger.info(f"Saved inference results to CSV ({csv_path}) and XLSX ({xlsx_path})")
        except Exception as e:
            logger.warning(
                "Failed to save XLSX file – check if openpyxl/xlsxwriter is installed.",
                error=str(e),
            )
            logger.info(f"Saved inference results to CSV only: {csv_path}")

        # Optionally log as W&B artifact
        artifact = wandb.Artifact(
            name="inference_table",
            type="dataset",
            description="Inference results table (CSV)",
        )
        artifact.add_file(str(csv_path))
        if xlsx_path.exists():
            artifact.add_file(str(xlsx_path))
        run.log_artifact(artifact)
    else:
        logger.warning("No rows collected for CSV/XLS export – nothing to save.")

    # ------------------------------------------------------------------
    # Save DEBUG stacked CSV
    # ------------------------------------------------------------------
    if debug_rows:
        import pandas as pd
        debug_df = pd.DataFrame(debug_rows)
        debug_csv_path = output_dir / f"inference_results_debug_{timestamp}.csv"
        debug_xlsx_path = output_dir / f"inference_results_debug_{timestamp}.xlsx"

        debug_df.to_csv(debug_csv_path, index=False, encoding="utf-8-sig")

        try:
            debug_df.to_excel(debug_xlsx_path, index=False)
        except Exception as e:
            logger.warning("Failed to save debug XLSX", error=str(e))

        # Log as separate artifact
        debug_artifact = wandb.Artifact(
            name="inference_table_debug",
            type="dataset",
            description="Inference results stacked by source (CSV/XLSX)",
        )
        debug_artifact.add_file(str(debug_csv_path))
        if debug_xlsx_path.exists():
            debug_artifact.add_file(str(debug_xlsx_path))
        run.log_artifact(debug_artifact)
    else:
        logger.warning("No debug rows collected – debug CSV not created.")
    
    logger.info("Inference completed successfully")
    run.finish()


if __name__ == "__main__":
    main() 