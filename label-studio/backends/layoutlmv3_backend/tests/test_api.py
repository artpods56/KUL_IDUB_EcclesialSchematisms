"""
This file contains tests for the API of your model. You can run these tests by installing test requirements:

    ```bash
    pip install -r requirements-test.txt
    ```
Then execute `pytest` in the directory of this file.

- Change `NewModel` to the name of the class in your model.py file.
- Change the `request` and `expected_response` variables to match the input and output of your model.
"""

import json
import os

import pytest
import logging
from model import LayoutLMv3Backend

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
@pytest.fixture(scope="session")
def model() -> LayoutLMv3Backend:
    model_checkpoint_path: str = os.getenv("DEFAULT_MODEL_CHECKPOINT", "checkpoint-400")
    return LayoutLMv3Backend(model_checkpoint_path)


@pytest.fixture
def client():
    import os

    from _wsgi import init_app
    from model import (
        LayoutLMv3Backend,
    )  # Ensure LayoutLMv3Backend is imported if not already

    # Get checkpoint path, default if not set
    checkpoint_path = os.getenv("DEFAULT_MODEL_CHECKPOINT", "checkpoint-400")

    app = init_app(
        model_class=LayoutLMv3Backend,
    )
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    # Create a temporary test image
    import os
    import tempfile

    from PIL import Image

    # Create a temporary image for testing
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        request = {
            "tasks": [{
                "id": 17,
                "data": {
                    "image": "/data/local-files/?d=wloclawek_1872/0017.jpg"
                },
                "meta": {},
                "created_at": "2025-04-27T02:07:17.316974Z", 
                "updated_at": "2025-04-27T12:14:53.509233Z",
                "is_labeled": True,
                "overlap": 1,
                "inner_id": 17,
                "total_annotations": 1,
                "cancelled_annotations": 0,
                "total_predictions": 0,
                "comment_count": 0,
                "unresolved_comment_count": 0,
                "last_comment_updated_at": None,
                "project": 1,
                "updated_by": 1,
                "file_upload": None,
                "comment_authors": [],
                "annotations": [{
                    "id": 2,
                    "result": [],
                    "created_username": "artpods56@gmail.com, 1",
                    "created_ago": "0 minutes",
                    "completed_by": 1,
                    "was_cancelled": False,
                    "ground_truth": False,
                    "created_at": "2025-04-27T12:14:53.478957Z",
                    "updated_at": "2025-04-27T12:14:53.478972Z", 
                    "draft_created_at": None,
                    "lead_time": 86.367,
                    "import_id": None,
                    "last_action": None,
                    "bulk_created": False,
                    "task": 17,
                    "project": 1,
                    "updated_by": 1,
                    "parent_prediction": None,
                    "parent_annotation": None,
                    "last_created_by": None
                }],
                "predictions": []
            }],
            "project": "1.1745719190",
            "label_config": "<View>\n  <Image name=\"image\" value=\"$image\"/>\n  <RectangleLabels name=\"label\" toName=\"image\">\n    \n    \n  <Label value=\"parish\" background=\"#FFA39E\"/><Label value=\"building_material\" background=\"#D4380D\"/><Label value=\"building_type\" background=\"#FFC069\"/><Label value=\"dedication\" background=\"#AD8B00\"/><Label value=\"settlement_classification\" background=\"#D3F261\"/><Label value=\"deanery\" background=\"#389E0D\"/><Label value=\"page_number\" background=\"#5CDBD3\"/></RectangleLabels>\n</View>",
            "params": {
                "login": None,
                "password": None,
                "context": None
            }
        }

        response = client.post(
            "/predict", data=json.dumps(request), content_type="application/json"
        )
        assert response.status_code == 200
        
        logger.info("Model response:",response.data)
        # Clean up
        os.unlink(temp_img.name)

        response_data = json.loads(response.data)
        assert "results" in response_data
        assert isinstance(response_data["results"], list)
