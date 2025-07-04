import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datasets import Dataset
from omegaconf import DictConfig

from llm.interface import LLMInterface
from dataset.parser import build_page_json
from dataset.stats import evaluate_entries
logger = logging.getLogger(__name__)


class DatasetEvaluator:
    """
    Handles dataset evaluation using LLM for predictions and optional LLM-as-a-judge evaluation.
    Separates prediction generation from evaluation logic.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize evaluator with separate configs for predictor and judge LLMs.
        
        Args:
            predictor_config: Configuration for the prediction LLM
            judge_config: Configuration for the judge LLM (if using LLM-as-a-judge)
            predictor_api_key: API key for predictor LLM
            judge_api_key: API key for judge LLM (can be same as predictor)
        """
        
        self.config = config
        self.predictor = LLMInterface(config.predictor)
        
        
        if config.judge.get("enabled", False):
            self.judge = LLMInterface(config.judge)
            
            

    def generate_prediction(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate prediction for a single dataset sample.
        
        Args:
            sample: Dataset sample with 'image' and other fields
            
        Returns:
            Parsed prediction as dictionary or None if failed
        """
        try:
            # Encode image
            base64_image = self.predictor.encode_image_to_base64(sample["image"])
            
            # Generate response
            response = self.predictor.generate_vision_response(
                base64_image=base64_image,
                system_prompt="system.j2",
                user_prompt="user.j2",
                context={}
            )
            
            if not response:
                logger.warning("No response generated for sample")
                return None
                
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None

    def evaluate_prediction(self, predictions, ground_truth):
        pass

    def evaluate_dataset(self, dataset: Dataset):

        
        predictions = []
        
        logger.info(f"Starting evaluation of {len(dataset)} samples")
        
        for i, sample in enumerate(dataset):
            logger.info(f"Processing sample {i+1}/{len(dataset)}: {sample.get(self.config.dataset.file_name_column_name, 'unknown')}")
            
            
            pil_image = sample.get(self.config.dataset.image_column_name, None)
            if not pil_image:
                logger.warning(f"No image found for sample {i+1}, skipping")
                continue
            
            words = sample.get(self.config.dataset.text_column_name, [])
            labels = sample.get(self.config.dataset.label_column_name, [])
            bboxes = sample.get(self.config.dataset.boxes_column_name, [])
            
            ground_truth_json = build_page_json(words, bboxes, labels)
            
            logger.info(f"Generated JSON for sample {i+1}: {ground_truth_json}")
            
            json_string_response = self.predictor.generate_vision_response(
                pil_image=pil_image,
                system_prompt="system.j2",
                user_prompt="user.j2",
            )
            
            json_prediction = json.loads(json_string_response)
            
            prediction_entries = json_prediction.get("entries", []) 
            ground_truth_entries = ground_truth_json.get("entries", [])
            tp, fp, fn = evaluate_entries(pred_entries=prediction_entries, gt_entries=ground_truth_entries)
            
            
            logger.info(f"Sample {i+1} evaluation: TP={tp}, FP={fp}, FN={fn}")