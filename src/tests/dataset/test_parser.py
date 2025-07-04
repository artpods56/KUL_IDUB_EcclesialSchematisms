import json

from dataset.parser import validate_bio_tags, bio_to_spans, build_page_json, standarize_building_material, normalize_field, strip_to_saint_core
import logging
from logger.setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TestBioParser:
    def test_validate_bio_tags(self, local_dataset_samples):

        labels = local_dataset_samples[0].get("labels", [])
        if not labels:
            raise ValueError("No labels found in the local dataset sample.")

        logger.info(f"Testing BIO tags validation for labels: {labels}")

        assert isinstance(labels, list), "Labels should be a list"

        issues = validate_bio_tags(labels)

        logger.info(f"Validation issues found: {issues}")
        assert not issues, "There should be no validation issues with the BIO tags"


    def test_complete_parsing(self, local_dataset_samples):
        """
        Test the complete parsing of a local dataset sample.
        """
        logger.info("Testing complete parsing of local dataset sample.")
        
        words = local_dataset_samples[0].get("words", [])
        bboxes = local_dataset_samples[0].get("bboxes", [])
        labels = local_dataset_samples[0].get("labels", [])

        
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

    
    
    
class TestTextNormalization:
    def test_standarize_building_material(self):
        """
        Test the building material standardization function.
        """
        test_cases = [
            ("mur.", "mur"),
            ("lig.", "lig"),
            ("null", "null"),
            ("murata", "mur"),
            ("ex murata", "mur"),
            ("ex mur.", "mur"),
            ("ex lig.", "lig"),
            ("lignea", "lig"),
        ]

        for input_material, expected_output in test_cases:
            standardized = standarize_building_material(input_material)
            assert standardized == expected_output, f"Expected {expected_output} but got {standardized} for input '{input_material}'"
            
    def test_field_standarization(self):
        
        
        test_cases = [
            ("S. Laurentius Martyr.", "laurentius"),
            ("S. Margaritha V. Mart.", "margaritha"),
        ]
        for case, gt in test_cases:

            assert strip_to_saint_core(case) == gt, f"Expected {gt} but got {strip_to_saint_core(case)} for input '{case}'"
            
            
    def test_normalize_field(self):
        """
        Test the field normalization function.
        """
        test_cases = [
            ("S. Laurentius Martyr.", "s. laurentius martyr."),
            ("S. Margaritha V. Mart.", "s. margaritha v. mart."),
            ("S. Maria Assumpta in Coelum", "s. maria assumpta in coelum"),
            ("S. Maria Assumpta in Coelo", "s. maria assumpta in coelo"),
        ]

        for input_field, expected_output in test_cases:
            normalized = normalize_field(input_field)
            assert normalized == expected_output, f"Expected {expected_output} but got {normalized} for input '{input_field}'"