import os
from PIL import Image
import argparse
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def cut_images_in_half(input_dir: str, output_dir: str, cut_vertically: bool = True):
    """
    Cuts all JPG images in a directory in half and saves them to an output directory.

    Args:
        input_dir (str): The directory containing the images to cut.
        output_dir (str): The directory to save the cut images to.
        cut_vertically (bool): If True, cut vertically. Otherwise, cut horizontally.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        files = [
            f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg"))
        ]
        if not files:
            logging.warning(f"No JPG or JPEG images found in {input_dir}")
            return
    except FileNotFoundError:
        logging.error(f"Input directory not found: {input_dir}")
        return

    logging.info(f"Found {len(files)} images to process.")

    for filename in tqdm(files, desc="Cutting images"):
        try:
            filepath = os.path.join(input_dir, filename)
            img = Image.open(filepath)
            width, height = img.size

            base, ext = os.path.splitext(filename)

            if cut_vertically:
                # Cut vertically
                midpoint = width // 2
                left_half = img.crop((0, 0, midpoint, height))
                right_half = img.crop((midpoint, 0, width, height))

                left_filename = f"{base}_1{ext}"
                right_filename = f"{base}_2{ext}"

                left_half.save(os.path.join(output_dir, left_filename))
                right_half.save(os.path.join(output_dir, right_filename))
            else:
                # Cut horizontally
                midpoint = height // 2
                top_half = img.crop((0, 0, width, midpoint))
                bottom_half = img.crop((0, midpoint, width, height))

                top_filename = f"{base}_1{ext}"
                bottom_filename = f"{base}_2{ext}"

                top_half.save(os.path.join(output_dir, top_filename))
                bottom_half.save(os.path.join(output_dir, bottom_filename))

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")


cut_images_in_half(input_dir="/tmp_bak/input", output_dir="/tmp_bak/input_cut")
