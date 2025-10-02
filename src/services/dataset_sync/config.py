"""
Configuration management for data sync and conversion scripts.
Loads configuration from .env file and merges with CLI arguments.
"""

import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv


def setup_logging(log_level: str):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(description)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )


def load_config_from_env(env_file: str) -> dict:
    """Load configuration from .env file and convert path strings to Path objects."""

    if env_file and Path(env_file).exists():
        print(f"Loading config from: {env_file}")
        load_dotenv(env_file)
    else:
        # If env_file is not found or not provided, we still proceed to load from system env
        # and use defaults. The `return` here was causing issues if env_file was None.
        print(f"Environment file '{env_file}' not found or not specified. Using system environment variables and defaults.")
        # No explicit load_dotenv() here means it relies on system env or pre-loaded ones.
    
    # Helper to get env var and convert to Path if it's a path key
    def get_path_env(key: str, default: str) -> Path:
        return Path(os.getenv(key, default))

    config = {
        # MinIO settings
        'MINIO_ENDPOINT': os.getenv('MINIO_ENDPOINT', 'localhost:9007'),
        'MINIO_ACCESS_KEY': os.getenv('MINIO_ROOT_USER'),
        'MINIO_SECRET_KEY': os.getenv('MINIO_ROOT_PASSWORD'),
        'MINIO_BUCKET': os.getenv('MINIO_BUCKET', 'annotations'),
        
        # HuggingFace settings
        'HF_REPO_URL': os.getenv('HF_REPO_URL'),
        'HF_REPO_DIR': get_path_env('HF_REPO_DIR', './hf_dataset'),
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        
        # Sync settings
        'POLL_INTERVAL': int(os.getenv('POLL_INTERVAL', '300')),
        'STATE_FILE': get_path_env('STATE_FILE', '.sync_state.json'), # Path object
        'MIN_NEW': int(os.getenv('MIN_NEW', '10')),
        
        # Conversion settings
        'CONVERT_SCRIPT': os.getenv('CONVERT_SCRIPT', 'convert_raw_annotations.py'), # This is a script description, not a path to manage here
        'IMAGE_DIR': get_path_env('IMAGE_DIR', './images'),
        'LS_ANNOTATIONS_DIR': get_path_env('LS_ANNOTATIONS_DIR', './label_studio_annotations'),
        'OUT_JSONL': get_path_env('OUT_JSONL', './train.jsonl'), # This will be a Path object
        'OCR_CACHE_DIR': get_path_env('OCR_CACHE_DIR', './ocr_cache'),
        
        # Logging
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
    }
    
    # This key is added by dataset_sync.py, not loaded from .env directly as a path
    # 'ENV_FILE': Path(env_file) if env_file else None, # Handled in dataset_sync.py

    print(f"Loaded config - MinIO endpoint: {config['MINIO_ENDPOINT']}, bucket: {config['MINIO_BUCKET']}")
    # Example of how a path is now a Path object
    if config['IMAGE_DIR']:
        print(f"Image directory (as Path object): {config['IMAGE_DIR']}, type: {type(config['IMAGE_DIR'])}")
    return config
