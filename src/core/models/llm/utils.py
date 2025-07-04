from io import BytesIO
import base64

def encode_image_to_base64(pil_image) -> str:
    """Convert PIL image to base64 string."""
    buffer = BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')