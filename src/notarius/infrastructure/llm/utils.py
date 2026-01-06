import base64
import hashlib
import io
from io import BytesIO
from pathlib import Path
from typing import Literal
import uuid
from PIL import Image

from notarius.domain.entities.messages import ChatMessage, TextContent, ImageContent


def parse_model_name(model_name: str) -> str:
    """Parse model name to ensure it is in a valid file format.

    Examples:
        /ml_models/gemma-3-27b-it-Q4_K_M.gguf -> gemma-3-27b-it-Q4_K_M
        gpt-4/turbo -> gpt-4_turbo
    """
    if Path(model_name).is_absolute():
        model_name = model_name.split("/")[-1].split(".")[0]

    return model_name.replace("/", "_")


def encode_image_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffer = BytesIO()
    if pil_image.mode == "RGB":
        pil_image.save(buffer, format="JPEG")
    else:
        converted = pil_image.convert("RGB")
        converted.save(buffer, format="JPEG")
        converted.close()
    result = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return result


def decode_base64_to_image(base64_url: str) -> Image.Image:
    """Decode data URI base64 string to PIL Image."""
    base64_data = base64_url.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_data)
    return Image.open(io.BytesIO(image_bytes))


def compute_image_hash(image: Image.Image, format: str = "JPEG") -> str:
    """Compute SHA-256 hash of image content."""
    buffer = io.BytesIO()
    img_to_hash = image if image.mode == "RGB" else image.convert("RGB")
    img_to_hash.save(
        buffer,
        format=format,
    )
    image_bytes = buffer.getvalue()
    return hashlib.sha256(image_bytes).hexdigest()


def generate_id() -> str:
    return str(uuid.uuid4())


def construct_text_message(
    text: str, role: Literal["user", "system", "developer", "assistant"]
) -> ChatMessage:
    """Construct text-only message from template using domain types."""

    return ChatMessage(
        role=role,
        content=[TextContent(text=text)],
    )


def construct_image_message(
    pil_image: Image.Image,
    text: str,
    role: Literal["user", "system", "developer", "assistant"],
    detail: Literal["auto", "low", "high"] = "auto",
) -> ChatMessage:
    """Construct multimodal message with image and text using domain types.

    Args:
        pil_image: PIL Image object
        text: Text prompt to accompany the image
        role: ChatMessage role
        detail: Image detail level for processing

    Returns:
        Domain ChatMessage with text and image content parts
    """
    # encode_image_to_base64 handles RGB conversion internally with proper cleanup
    base64_image = encode_image_to_base64(pil_image)
    image_url = f"data:image/jpeg;base64,{base64_image}"

    return ChatMessage(
        role=role,
        content=[
            TextContent(text=text),
            ImageContent(image_url=image_url, detail=detail),
        ],
    )
