import base64
from io import BytesIO

from openai.types.chat import ChatCompletionMessageParam


# ... (all your other imports)


def make_all_properties_required(schema: dict) -> dict:
    """
    Recursively modifies a JSON schema to make all properties required,
    to comply with strict API validation rules (like some Azure OpenAI endpoints).
    """
    if "properties" in schema and isinstance(schema["properties"], dict):
        # Make all properties in the current level required
        schema["required"] = list(schema["properties"].keys())

        # Recurse into nested properties (for nested objects)
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict):
                make_all_properties_required(prop_schema)

    # Recurse into array items (for lists of objects)
    if "items" in schema and isinstance(schema["items"], dict):
        make_all_properties_required(schema["items"])

    # Recurse into $defs (for referenced schemas)
    if "$defs" in schema and isinstance(schema["$defs"], dict):
        for def_name, def_schema in schema["$defs"].items():
            if isinstance(def_schema, dict):
                make_all_properties_required(def_schema)

    return schema


def encode_image_to_base64(pil_image) -> str:
    """Convert PIL image to base64 string."""
    buffer = BytesIO()
    pil_image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def messages_to_string(messages: list[ChatCompletionMessageParam]) -> str:
    result = []

    for message in messages:
        role = message.get("role", "unknown")

        # Handle different content types
        content = message.get("content", "")

        if isinstance(content, str):
            # Simple string content
            result.append(f"{role}: {content}")
        elif isinstance(content, list):
            # List of content parts (text + images)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                # Skip image parts (type == 'image_url')
            if text_parts:
                result.append(f"{role}: {' '.join(text_parts)}")

    return "\n\n".join(result)
