"""
Pytest configuration for OCR API tests.
"""

import pytest
import os
from pathlib import Path

@pytest.fixture(autouse=True)
def test_env():
    """Set up test environment variables for all tests."""
    # Store original environment variables
    original_env = {}
    for var in ['MODEL_PATH', 'IMAGE_DIR', 'OUTPUT_DIR']:
        if var in os.environ:
            original_env[var] = os.environ[var]
    
    # Set up test environment variables
    os.environ['MODEL_PATH'] = 'ds4sd/SmolDocling-256M-preview'
    os.environ['IMAGE_DIR'] = str(Path(__file__).parent / 'test_img')
    os.environ['OUTPUT_DIR'] = str(Path(__file__).parent / 'test_out')
    
    # Create test directories
    os.makedirs(os.environ['IMAGE_DIR'], exist_ok=True)
    os.makedirs(os.environ['OUTPUT_DIR'], exist_ok=True)
    
    yield
    
    # Restore original environment variables
    for var in ['MODEL_PATH', 'IMAGE_DIR', 'OUTPUT_DIR']:
        if var in original_env:
            os.environ[var] = original_env[var]
        else:
            os.environ.pop(var, None)