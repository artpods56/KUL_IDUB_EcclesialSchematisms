"""
Pytest configuration for Streamlit app tests.
"""

import pytest
import os

@pytest.fixture(autouse=True)
def test_env():
    """Set up test environment variables for all tests."""
    # Store original environment variables
    original_env = {}
    for var in ['OCR_API_URL']:
        if var in os.environ:
            original_env[var] = os.environ[var]
    
    # Set up test environment variables
    os.environ['OCR_API_URL'] = 'http://test-api:8000'
    
    yield
    
    # Restore original environment variables
    for var in ['OCR_API_URL']:
        if var in original_env:
            os.environ[var] = original_env[var]
        else:
            os.environ.pop(var, None)