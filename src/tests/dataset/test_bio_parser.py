import json

from dataset.parser import validate_bio_tags, bio_to_spans, build_page_json
import logging
from logger.setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TestBioParser:
    def test_validate_bio_tags(self, local_dataset_sample):

        labels = local_dataset_sample.get("labels", [])
        if not labels:
            raise ValueError("No labels found in the local dataset sample.")

        logger.info(f"Testing BIO tags validation for labels: {labels}")

        assert isinstance(labels, list), "Labels should be a list"

        issues = validate_bio_tags(labels)

        logger.info(f"Validation issues found: {issues}")
        assert not issues, "There should be no validation issues with the BIO tags"


    def test_complete_parsing(self, local_dataset_sample):
        """
        Test the complete parsing of a local dataset sample.
        """
        logger.info("Testing complete parsing of local dataset sample.")
        
        words = local_dataset_sample.get("words", [])
        bboxes = local_dataset_sample.get("bboxes", [])
        labels = local_dataset_sample.get("labels", [])

        
        result = build_page_json(words, bboxes, labels)

        logger.info(json.dumps(result, indent=2, ensure_ascii=False))


    def test_complete_parsing_on_batch(self, cfg, local_dataset_samples):
        """
        Test the complete parsing of a dataset sample.
        """

        for sample in local_dataset_samples:
            image_name = sample.get(cfg.dataset.file_name_column_name, "unknown_image")

            logger.info(f"Processing image: {image_name}")

            words = sample.get(cfg.dataset.text_column_name, [])
            bboxes = sample.get(cfg.dataset.boxes_column_name, [])
            labels = sample.get(cfg.dataset.label_column_name, [])

            result = build_page_json(words, bboxes, labels)

            logger.info(json.dumps(result, indent=2, ensure_ascii=False))

    