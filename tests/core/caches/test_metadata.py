from pathlib import Path
from PIL import Image
import pytest

from core.caches.base_cache import PredictMetadata
from core.caches.llm_cache import LLMCache


def test_metadata_storage_and_retrieval(tmp_path: Path):
    """Test that metadata is properly stored and retrieved from cache."""
    
    cache_root = tmp_path / "cache_root"
    cache = LLMCache(model_name="test-model", caches_dir=cache_root)
    
    # Create test metadata
    metadata = PredictMetadata(
        schematism="wloclawek_1872",
        filename="0005.jpg",
        model_name="gpt-4o",
        inference_type="vision",
        dataset_split="test",
        experiment_name="test_experiment"
    )
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='white')
    
    # Generate cache key
    key = cache.generate_hash(
        image_hash=cache.get_image_hash(test_image),
        text_hash=None,
        hints=None,
        messages=[]
    )
    
    # Store cache item with metadata
    cache_item_data = {
        "response": {"test": "response"},
        "hints": None,
        "metadata": metadata.model_dump(),
        "schematism": metadata.schematism
    }
    
    cache[key] = cache_item_data
    
    # Retrieve and verify metadata
    retrieved_item = cache[key]
    assert retrieved_item["response"]["test"] == "response"
    assert "metadata" in retrieved_item
    
    # Verify metadata fields
    stored_metadata = retrieved_item["metadata"]
    assert stored_metadata["schematism"] == "wloclawek_1872"
    assert stored_metadata["filename"] == "0005.jpg"
    assert stored_metadata["model_name"] == "gpt-4o"
    assert stored_metadata["inference_type"] == "vision"
    assert stored_metadata["dataset_split"] == "test"
    assert stored_metadata["experiment_name"] == "test_experiment"


def test_metadata_backward_compatibility(tmp_path: Path):
    """Test that cache works without metadata (backward compatibility)."""
    
    cache_root = tmp_path / "cache_root"
    cache = LLMCache(model_name="test-model", caches_dir=cache_root)
    
    # Store cache item without metadata
    key = cache.generate_hash(image_hash=None, text_hash="123", hints=None, messages=[])
    cache_item_data = {
        "response": {"test": "response"},
        "hints": None
    }
    
    cache[key] = cache_item_data
    
    # Retrieve and verify it works
    retrieved_item = cache[key]
    assert retrieved_item["response"]["test"] == "response"
    assert "metadata" not in retrieved_item


def test_metadata_optional_fields(tmp_path: Path):
    """Test that optional metadata fields work correctly."""
    
    cache_root = tmp_path / "cache_root"
    cache = LLMCache(model_name="test-model", caches_dir=cache_root)
    
    # Create minimal metadata with only required fields
    metadata = PredictMetadata(
        schematism="wloclawek_1872",
        filename="0005.jpg"
    )
    
    key = cache.generate_hash(image_hash=None, text_hash="123", hints=None, messages=[])
    cache_item_data = {
        "response": {"test": "response"},
        "hints": None,
        "metadata": metadata.model_dump(),
        "schematism": metadata.schematism
    }
    
    cache[key] = cache_item_data
    
    # Retrieve and verify
    retrieved_item = cache[key]
    stored_metadata = retrieved_item["metadata"]
    assert stored_metadata["schematism"] == "wloclawek_1872"
    assert stored_metadata["filename"] == "0005.jpg"
    assert stored_metadata["model_name"] is None  # Optional field should be None 