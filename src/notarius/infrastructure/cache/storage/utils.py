import hashlib

from PIL import Image

from notarius.domain.entities.messages import ChatMessage, ImageContent
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.persistence.storage import ImageRepository

from notarius.infrastructure.llm.utils import (
    compute_image_hash,
    decode_base64_to_image,
)
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


def get_base64_hash(base64_str: str) -> str:
    """Generate a SHA-256 hash for a base64 string.

    Args:
        base64_str: Base64 encoded string (optionally with data: prefix)

    Returns:
        SHA-256 hash of the base64 content
    """
    if "," in base64_str:
        # Strip data:image/jpeg;base64, prefix if present
        base64_str = base64_str.split(",", 1)[1]
    return hashlib.sha256(base64_str.encode()).hexdigest()


def replace_images_with_refs(
    message: ChatMessage,
    image_repository: ImageRepository,
) -> ChatMessage:
    """Replace base64 images with content-addressable references.

    Args:
        message: ChatMessage potentially containing base64 images
        image_repository: Repository for storing images by content hash

    Returns:
        New ChatMessage with images replaced by references
    """


    new_content = []

    for part in message.content:
        if isinstance(part, ImageContent) and not part.image_url.startswith("ref://"):
            # Decode and hash the image
            pil_image = decode_base64_to_image(part.image_url)
            image_hash = compute_image_hash(pil_image)

            # Store if not exists (automatic deduplication)
            if not image_repository.exists(image_hash):
                image_repository.add(pil_image, name=image_hash)

            # Replace with reference
            new_content.append(
                ImageContent(image_url=f"ref://{image_hash}", detail=part.detail)
            )
        else:
            new_content.append(part)

    return ChatMessage(role=message.role, content=new_content)


def resolve_image_refs(
    message: ChatMessage,
    image_repository: ImageRepository,
) -> ChatMessage:
    """Resolve image references back to base64 data URIs.

    Args:
        message: ChatMessage with image references
        image_repository: Repository for loading images by hash

    Returns:
        New ChatMessage with base64 images restored
    """
    from notarius.infrastructure.llm.utils import encode_image_to_base64

    new_content = []

    for part in message.content:
        if isinstance(part, ImageContent) and part.image_url.startswith("ref://"):
            # Extract hash and load image
            image_hash = part.image_url.replace("ref://", "")
            image_path = image_repository.get_path(image_hash)
            pil_image = image_repository.get(image_path)

            # Encode to base64
            base64_image = encode_image_to_base64(pil_image)
            base64_url = f"data:image/jpeg;base64,{base64_image}"

            new_content.append(
                ImageContent(image_url=base64_url, detail=part.detail)
            )
        else:
            new_content.append(part)

    return ChatMessage(role=message.role, content=new_content)


def conversation_with_refs(
    conversation: Conversation,
    image_repository: ImageRepository
) -> Conversation:
    """Convert all conversation images to refs.

    Args:
        conversation: Conversation with base64 images
        image_repository: Repository for storing images

    Returns:
        New Conversation with all images replaced by references
    """
    new_messages = [
        replace_images_with_refs(msg, image_repository)
        for msg in conversation.messages
    ]
    return Conversation(messages=tuple(new_messages))


def conversation_with_images(
    conversation: Conversation,
    image_repository: ImageRepository
) -> Conversation:
    """Resolve all conversation refs to images.

    Args:
        conversation: Conversation with image references
        image_repository: Repository for loading images

    Returns:
        New Conversation with all references resolved to base64 images
    """
    new_messages = [
        resolve_image_refs(msg, image_repository) for msg in conversation.messages
    ]
    return Conversation(messages=tuple(new_messages))
