from PIL import Image, ImageDraw, ImageFont
import json

def unnormalize_bbox(bbox, image_width, image_height):
    """
    Convert normalized bbox [0-1000] back to pixel coordinates.
    
    Args:
        bbox: [x1, y1, x2, y2] in normalized coordinates (0-1000)
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
    
    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    return [
        bbox[0] * image_width / 1000,
        bbox[1] * image_height / 1000,
        bbox[2] * image_width / 1000,
        bbox[3] * image_height / 1000
    ]

def visualize_bboxes(image, words, bboxes, labels, output_path=None):
    """
    Simple function to draw bounding boxes on an image.
    
    Args:
        image: PIL Image object
        words: List of words/tokens
        bboxes: List of normalized bboxes [0-1000 scale]
        labels: List of labels
        output_path: Optional path to save the image
    
    Returns:
        PIL Image with drawn bboxes
    """
    # Convert to RGB if needed
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("Arial", 10)
    except:
        font = ImageDraw.getfont()
    
    # Color mapping for different labels
    colors = {
        'B-parish': 'red',
        'I-parish': 'darkred',
        'B-dedication': 'blue', 
        'I-dedication': 'darkblue',
        'B-building_material': 'green',
        'I-building_material': 'darkgreen',
        'B-page_number': 'purple',
        'I-page_number': 'darkpurple',
        'O': 'gray'
    }
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Draw each bbox
    for word, bbox, label in zip(words, bboxes, labels):
        # Skip 'O' labels if you want to focus on entities only
        if label == 'O':
            continue
            
        # Convert normalized bbox to pixel coordinates
        pixel_bbox = unnormalize_bbox(bbox, img_width, img_height)
        
        # Get color for this label
        color = colors.get(label, 'yellow')
        
        # Draw the bounding box
        draw.rectangle(pixel_bbox, outline=color, width=2)
        
        # Draw label text above the box
        text = f"{label}: {word}"
        text_y = max(0, pixel_bbox[1] - 15)
        
        # Draw text with white background for visibility
        text_bbox = draw.textbbox((pixel_bbox[0], text_y), text, font=font)
        draw.rectangle(text_bbox, fill='white')
        draw.text((pixel_bbox[0], text_y), text, fill=color, font=font)
    
    # Save if output path provided
    if output_path:
        image.save(output_path)
    
    return image

# Example usage with your data:
