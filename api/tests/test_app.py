"""
Tests for the FastAPI app for SmolDocling OCR.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import io

# Import our application
from app import app

# Test client
client = TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def mock_llm():
    """Mock the LLM class."""
    with patch("app.LLM") as mock:
        # Mock the generate method
        mock_instance = MagicMock()
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "<doc>This is a test document</doc>"
        mock_instance.generate.return_value = [mock_output]
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def mock_docling():
    """Mock Docling document classes."""
    with patch("app.DocTagsDocument") as mock_tags_doc, \
         patch("app.DoclingDocument") as mock_doc:
        # Configure the mocks
        mock_tags_doc.from_doctags_and_image_pairs.return_value = MagicMock()
        mock_doc_instance = MagicMock()
        mock_doc.return_value = mock_doc_instance
        
        yield {
            "tags_doc": mock_tags_doc,
            "doc": mock_doc,
            "doc_instance": mock_doc_instance
        }


def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "model" in response.json()


@patch("app.llm")
def test_list_outputs(mock_llm, tmp_path):
    """Test listing output files."""
    # Create a temporary output directory with files
    original_output_dir = app.OUTPUT_DIR
    app.OUTPUT_DIR = str(tmp_path)
    
    # Create some test files
    test_files = ["test1.md", "test2.dt"]
    for file in test_files:
        with open(os.path.join(tmp_path, file), "w") as f:
            f.write("test content")
    
    # Test the endpoint
    response = client.get("/outputs/")
    assert response.status_code == 200
    assert "files" in response.json()
    assert set(response.json()["files"]) == set(test_files)
    
    # Restore the original directory
    app.OUTPUT_DIR = original_output_dir


@patch("app.llm")
def test_get_output_file_not_found(mock_llm):
    """Test getting a non-existent output file."""
    response = client.get("/outputs/nonexistent.md")
    assert response.status_code == 404


@patch("app.os.path.exists")
@patch("app.FileResponse")
@patch("app.llm")
def test_get_output_file_exists(mock_llm, mock_file_response, mock_exists):
    """Test getting an existing output file."""
    # Mock that the file exists
    mock_exists.return_value = True
    mock_file_response.return_value = "file content"
    
    response = client.get("/outputs/test.md")
    assert response != "file content"  # We don't get the actual content in the test client
    mock_file_response.assert_called_once()


@pytest.mark.asyncio
async def test_process_image(mock_llm, mock_docling, sample_image, tmp_path):
    """Test the image processing endpoint."""
    # Setup temporary directories
    original_image_dir = app.IMAGE_DIR
    original_output_dir = app.OUTPUT_DIR
    app.IMAGE_DIR = str(tmp_path / "img")
    app.OUTPUT_DIR = str(tmp_path / "out")
    os.makedirs(app.IMAGE_DIR, exist_ok=True)
    os.makedirs(app.OUTPUT_DIR, exist_ok=True)
    
    # Initialize the LLM
    app.llm = mock_llm.return_value
    
    # Test the endpoint with our sample image
    response = client.post(
        "/ocr/",
        files={"file": ("test_image.jpg", sample_image, "image/jpeg")}
    )
    
    # Check response
    assert response.status_code == 200
    result = response.json()
    assert "filename" in result
    assert "markdown_path" in result
    assert "doctags_path" in result
    assert "processing_time" in result
    
    # Verify calls to mocks
    assert mock_llm.return_value.generate.called
    assert mock_docling["tags_doc"].from_doctags_and_image_pairs.called
    assert mock_docling["doc_instance"].load_from_doctags.called
    assert mock_docling["doc_instance"].save_as_markdown.called
    
    # Restore original directories
    app.IMAGE_DIR = original_image_dir
    app.OUTPUT_DIR = original_output_dir


def test_process_image_invalid_file():
    """Test uploading an invalid file type."""
    response = client.post(
        "/ocr/",
        files={"file": ("test.txt", b"This is not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]