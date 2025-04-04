"""
Tests for the Streamlit app for SmolDocling OCR.
"""

import os
import io
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

# Since Streamlit apps are not directly testable as modules,
# we test the underlying functions in the app.py file
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now we can import functions from app.py
from app import check_api_status, process_image, fetch_output_content


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


def test_check_api_status_success():
    """Test API status check when API is available."""
    with patch('requests.get') as mock_get:
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'model': 'test-model'}
        mock_get.return_value = mock_response
        
        available, message = check_api_status()
        
        assert available is True
        assert 'test-model' in message
        mock_get.assert_called_once()


def test_check_api_status_error():
    """Test API status check when API is not available."""
    with patch('requests.get') as mock_get:
        # Mock connection error
        mock_get.side_effect = Exception("Connection error")
        
        available, message = check_api_status()
        
        assert available is False
        assert 'Connection error' in message


def test_process_image_success(sample_image):
    """Test successful image processing."""
    with patch('requests.post') as mock_post:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'filename': 'test.jpg',
            'markdown_path': '/app/out/test.md',
            'doctags_path': '/app/out/test.dt',
            'processing_time': 1.23
        }
        mock_post.return_value = mock_response
        
        # Mock streamlit's error function
        with patch('streamlit.error') as mock_st_error:
            result = process_image(sample_image, 'test.jpg')
            
            assert result == mock_response.json.return_value
            mock_post.assert_called_once()
            mock_st_error.assert_not_called()


def test_process_image_api_error():
    """Test image processing when API returns an error."""
    with patch('requests.post') as mock_post:
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response
        
        # Mock streamlit's error function
        with patch('streamlit.error') as mock_st_error:
            result = process_image(b'test data', 'test.jpg')
            
            assert result is None
            mock_st_error.assert_called_once()


def test_process_image_connection_error():
    """Test image processing when connection to API fails."""
    with patch('requests.post') as mock_post:
        # Mock connection error
        mock_post.side_effect = Exception("Connection error")
        
        # Mock streamlit's error function
        with patch('streamlit.error') as mock_st_error:
            result = process_image(b'test data', 'test.jpg')
            
            assert result is None
            mock_st_error.assert_called_once()


def test_fetch_output_content_success():
    """Test successful output content fetching."""
    with patch('requests.get') as mock_get:
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Test Document"
        mock_get.return_value = mock_response
        
        content = fetch_output_content('/app/out/test.md')
        
        assert content == "# Test Document"
        mock_get.assert_called_once()
        

def test_fetch_output_content_error():
    """Test output content fetching when API returns an error."""
    with patch('requests.get') as mock_get:
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        content = fetch_output_content('/app/out/test.md')
        
        assert content is None
        mock_get.assert_called_once()