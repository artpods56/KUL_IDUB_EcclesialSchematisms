"""
Script to test the OCR API by uploading a test image.
"""

import argparse
import os
import requests
import time
import json
from pathlib import Path

def test_api(
    api_url: str,
    image_path: str,
    verbose: bool = False
):
    """
    Test the OCR API by uploading a test image.
    
    Args:
        api_url: Base URL of the OCR API
        image_path: Path to the image file to upload
        verbose: Whether to print detailed information
    """
    # Check if API is running
    try:
        start_time = time.time()
        response = requests.get(f"{api_url}/")
        if response.status_code == 200:
            print(f"✓ API is running: {response.json()}")
        else:
            print(f"✗ API responded with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Error connecting to API: {str(e)}")
        return False
    
    # Upload image for OCR processing
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": (os.path.basename(image_path), img_file, "image/jpeg")}
            print(f"Uploading image: {image_path}")
            response = requests.post(f"{api_url}/ocr/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                elapsed = time.time() - start_time
                print(f"✓ OCR processing successful in {elapsed:.2f}s (API processing: {result.get('processing_time', 0):.2f}s)")
                
                if verbose:
                    print(f"Result: {json.dumps(result, indent=2)}")
                
                # Try to fetch the markdown output
                md_filename = os.path.basename(result["markdown_path"])
                md_response = requests.get(f"{api_url}/outputs/{md_filename}")
                
                if md_response.status_code == 200:
                    print("\nExtracted text sample:")
                    print("-" * 40)
                    # Print first few lines
                    lines = md_response.text.split("\n")
                    print("\n".join(lines[:10]))
                    print("-" * 40)
                else:
                    print(f"✗ Could not fetch markdown content: {md_response.status_code}")
                
                return True
            else:
                print(f"✗ OCR processing failed: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"✗ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the OCR API")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="Base URL of the OCR API")
    parser.add_argument("--image", type=str, default=None, help="Path to image file to upload")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    # If no image specified, check if there's any in the img directory
    if args.image is None:
        img_dir = Path(__file__).parent.parent / "img"
        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        if image_files:
            args.image = str(image_files[0])
            print(f"No image specified, using: {args.image}")
        else:
            print("No image file found. Please specify an image with --image or add an image to the img directory.")
            exit(1)
    
    success = test_api(
        api_url=args.api_url,
        image_path=args.image,
        verbose=args.verbose
    )
    
    exit(0 if success else 1)