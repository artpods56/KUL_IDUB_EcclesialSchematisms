"""
Configuration management for dataset sync and conversion scripts.
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )


def load_config_from_env(env_file: str) -> dict:
    """Load configuration from .env file."""

    root_dir = Path(__file__).parent.parent
    env_path = root_dir / env_file
    
    data_dir = root_dir / 'data'
    
    if not env_path.exists():
        return False
    
    if env_path.exists():
        load_dotenv(env_path)
    
    config = {
        'MINIO_ENDPOINT': os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
        'MINIO_ACCESS_KEY': os.getenv('MINIO_ROOT_USER'),  # Using your existing name
        'MINIO_SECRET_KEY': os.getenv('MINIO_ROOT_PASSWORD'),  # Using your existing name
        'MINIO_BUCKET': os.getenv('MINIO_BUCKET', 'schematyzmy'),  # Default bucket name
        
        'HF_REPO_URL': os.getenv('HF_REPO_URL'),
        'HF_REPO_DIR': data_dir / Path(str(os.getenv('HF_REPO_DIR'))),
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        
        'POLL_INTERVAL': int(os.getenv('POLL_INTERVAL', '300')),
        'STATE_FILE': data_dir / Path(os.getenv('STATE_FILE', '.sync_state.json')),
        'MIN_NEW': int(os.getenv('MIN_NEW', '10')),
        
        'CONVERT_SCRIPT': os.getenv('CONVERT_SCRIPT', 'convert_raw_annotations.py'),
        'IMAGE_DIR': data_dir / Path(str(os.getenv('IMAGE_DIR'))),
        'LS_ANNOTATIONS_DIR': data_dir / Path(str(os.getenv('LS_ANNOTATIONS_DIR'))),
        'OUT_JSONL': Path(str(os.getenv('OUT_JSONL'))),
        'OCR_CACHE_DIR': data_dir / Path(str(os.getenv('OCR_CACHE_DIR'))),
        
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'DATA_DIR': data_dir,
    }
    
    return config
