from PIL import Image
import hashlib

def get_image_hash(pil_image: Image.Image) -> str:
    """Generate a hash for the image to use as cache key."""
    # Convert to bytes for hashing
    img_bytes = pil_image.tobytes()
    return hashlib.md5(img_bytes).hexdigest()


def get_text_hash(text: str | None) -> str | None:
    """Return a deterministic SHA-256 hash for *text*.

    Returns None if *text* is ``None`` so callers can pass the value
    directly into ``generate_hash`` without extra conditionals.
    """
    if text is None:
        return None
    return hashlib.sha256(text.encode()).hexdigest()
