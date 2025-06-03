import json
import os
from pathlib import Path
import datasets
from PIL import Image
import ast

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{,
  title={},
  author={},
  journal={},
  year={},
  volume={}
}
"""
_DESCRIPTION = """\
This is a sample dataset for training layoutlmv3 model on custom annotated data.
"""

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w,h)

_URLS = []

# Use absolute path to images directory
data_path = '/Users/user/Projects/ecclesiasticalOCR/layoutLMv3/images'

class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for SchematismExtraction Dataset"""
    def __init__(self, **kwargs):
        """BuilderConfig for SchematismExtraction Dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(**kwargs)


class SchematismExtraction(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        DatasetConfig(name="SchematismExtraction", version=datasets.Version("1.0.0"), description="SchematismsExtraction dataset"),
    ]

    def _info(self):
        labels = ['page_number', 'building_material', 'settlement_classification', 'parish', 'building_type', 'dedication', 'deanery']
        bio_labels = ["B-{}".format(label) for label in labels] + ["I-{}".format(label) for label in labels] + ["O"]

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
            {
                "id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                "ner_tags": datasets.Sequence(
                datasets.features.ClassLabel(
                    names=bio_labels,
                    )
                ),
                "image_path": datasets.Value("string"),
                "image": datasets.features.Image()
            }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_path, "train.txt"),
                    "dest": data_path
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_path, "test.txt"),
                    "dest": data_path
                }
            ),
        ]

    def _generate_examples(self, filepath, dest):
        try:
            logger.info("⏳ Generating examples from = %s", filepath)
            
            item_list = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    item_list.append(line.rstrip('\n\r'))

            for guid, fname in enumerate(item_list):
                try:
                    data = ast.literal_eval(fname)
                    image_path = data["image_path"]
                    try:
                        image, size = load_image(image_path)
                        # If no labels exist, create empty lists
                        text = data.get('tokens', [])
                        boxes = data.get('bboxes', [])
                        label = data.get('ner_tags', [])
                        # Only validate lengths if we have data
                        if text and (len(text) != len(boxes) or len(text) != len(label)):
                            logger.warning(f"Skipping example {guid} due to mismatched lengths: tokens={len(text)}, boxes={len(boxes)}, labels={len(label)}")
                            continue
                        yield guid, {
                            "id": str(guid),
                            "tokens": text,
                            "bboxes": boxes,
                            "ner_tags": label,
                            "image_path": image_path,
                            "image": image
                        }
                    except Exception as e:
                        logger.error(f"Error processing example {guid}: {str(e)}")
                        continue
                except Exception as e:
                    logger.error(f"Error parsing example {guid}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            raise
