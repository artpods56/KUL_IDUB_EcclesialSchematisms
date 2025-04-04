"""
Creates a test image with text for OCR testing.
"""

import argparse
import os
from PIL import Image, ImageDraw, ImageFont

def create_test_image(
    output_path: str,
    text: str = "This is a test document for OCR.",
    width: int = 800,
    height: int = 600,
    bg_color: str = "white",
    text_color: str = "black",
):
    """
    Creates a test image with the specified text.
    
    Args:
        output_path: Path where the image will be saved
        text: Text to write on the image
        width: Image width in pixels
        height: Image height in pixels
        bg_color: Background color
        text_color: Text color
    """
    # Create a blank image
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a font
    try:
        # Try to use a default font if available
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except IOError:
        # Use default font if the specific font is not available
        font = ImageFont.load_default()
    
    # Calculate position to center the text
    text_width, text_height = draw.textsize(text, font=font)
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw text on the image
    draw.text(position, text, font=font, fill=text_color)
    
    # Save the image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"Test image created at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a test image with text for OCR testing")
    parser.add_argument("--output", type=str, default="img/test_image.jpg", help="Output image path")
    parser.add_argument("--text", type=str, default="This is a test document for OCR.", help="Text to write on the image")
    parser.add_argument("--width", type=int, default=800, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=600, help="Image height in pixels")
    
    args = parser.parse_args()
    
    create_test_image(
        output_path=args.output,
        text=args.text,
        width=args.width,
        height=args.height
    )