# tests/conftest.py
from typing import cast, List

import pytest
from datasets import IterableDataset
from hydra import compose, initialize_config_dir
from dataset.utils import get_dataset
from pathlib import Path
import os
import random
import json

import time

from shared import PROJECT_ROOT, CONFIGS_DIR

TESTS_DIR = Path(__file__).resolve().parent / "tests"
SAMPLE_DATA_DIR = TESTS_DIR / "sample_data"

@pytest.fixture(scope="session")
def cfg():
    tests_config_dir = CONFIGS_DIR / "tests"
    with initialize_config_dir(config_dir=str(tests_config_dir), job_name="test", version_base="1.1"):
        return compose(config_name="config")

@pytest.fixture(scope="session")
def dataset_sample(cfg):
    """
    Fixture to load the dataset based on the provided configuration.
    """
    dataset = get_dataset(cfg)
    dataset.shuffle(seed=cfg.dataset.seed)

    if dataset is None:
        raise ValueError("Dataset could not be loaded. Please check the configuration.")
    
    return next(iter(dataset))

@pytest.fixture(scope="session")
def local_dataset_samples() -> List[dict]:
    """Returns a sample from the local dataset."""
    local_dataset_path = SAMPLE_DATA_DIR / Path("dataset_sample.jsonl")
    samples = []
    with open(local_dataset_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    if not samples:
        raise ValueError("Local dataset samples are empty. Please check the file content.")
    
    return samples

@pytest.fixture(scope="session")
def random_dataset_sample(cfg):
    """
    Fixture to load the dataset based on the provided configuration.
    """
    dataset = cast(IterableDataset, get_dataset(cfg))

    if dataset is None:
        raise ValueError("Dataset could not be loaded. Please check the configuration.")
    
    buffer = 50          # larger buffer → better randomness, more RAM
    seed   = int(time.time())  # change seed each call if you want a new sample

    return next(iter(dataset.shuffle(seed=seed, buffer_size=buffer).take(1)))
