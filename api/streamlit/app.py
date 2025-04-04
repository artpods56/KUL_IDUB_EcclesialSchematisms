"""
Streamlit interface for OCR service using SmolDocling.
"""

import os
import io
import time
import requests
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from dotenv import load_dotenv
import streamlit as st
from PIL import Image

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.environ.get("OCR_API_URL", "http://ocr-api:8000")

# Page setup
st.set_page_config(
    page_title="Ecclesiastical OCR",
    page_icon="📚",
    layout="wide",
)

# Title and description
st.title("Ecclesiastical OCR")
st.markdown("""
This application helps you convert scanned ecclesiastical documents into structured text using OCR (Optical Character Recognition).
Upload a document image, and it will be processed using SmolDocling OCR technology.
""")

# Functions
def check_api_status() -> Tuple[bool, str]:
    """
    Check if the OCR API is available.
    
    Returns:
        Tuple[bool, str]: (is_available, message)
    """
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, f"API available using model: {data.get('model', 'unknown')}"
        return False, f"API responded with status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Error connecting to API: {str(e)}"

def process_image(file_data: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """
    Send image to OCR API for processing.
    
    Args:
        file_data: The image file data
        filename: Original filename
        
    Returns:
        Optional[Dict[str, Any]]: API response or None if failed
    """
    try:
        files = {"file": (filename, file_data, "image/jpeg")}
        response = requests.post(f"{API_URL}/ocr/", files=files, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {str(e)}")
        return None

def fetch_output_content(path: str) -> Optional[str]:
    """
    Fetch content of an output file from the API.
    
    Args:
        path: Path to the file
        
    Returns:
        Optional[str]: File content or None if failed
    """
    filename = os.path.basename(path)
    try:
        response = requests.get(f"{API_URL}/outputs/{filename}", timeout=10)
        if response.status_code == 200:
            return response.text
        return None
    except requests.exceptions.RequestException:
        return None

# Main application
def main():
    # Show API status
    api_available, api_message = check_api_status()
    status_color = "green" if api_available else "red"
    st.markdown(f"<p style='color:{status_color};'>API Status: {api_message}</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Options")
    st.sidebar.markdown("Configure OCR processing options:")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # Process button
        if st.button("Process Image", disabled=(not api_available)):
            if not api_available:
                st.error("Cannot process image: API is not available")
            else:
                with st.spinner("Processing image with OCR..."):
                    # Reset file pointer and read data
                    uploaded_file.seek(0)
                    file_data = uploaded_file.read()
                    
                    # Process image
                    start_time = time.time()
                    result = process_image(file_data, uploaded_file.name)
                    
                    if result:
                        processing_time = time.time() - start_time
                        st.success(f"✅ Image processed in {processing_time:.2f} seconds (API processing: {result.get('processing_time', 0):.2f}s)")
                        
                        # Fetch markdown content
                        with col2:
                            st.subheader("Extracted Text")
                            markdown_content = fetch_output_content(result["markdown_path"])
                            if markdown_content:
                                st.markdown(markdown_content)
                            else:
                                st.warning("Could not fetch markdown content")
                        
                        # Show raw doctags in expander
                        with st.expander("View Raw DocTags"):
                            doctags_content = fetch_output_content(result["doctags_path"])
                            if doctags_content:
                                st.code(doctags_content)
                            else:
                                st.warning("Could not fetch doctags content")
    
    # Information section
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This application uses SmolDocling, a specialized OCR model for document processing.
    The system extracts text from scanned pages and converts it to structured formats.
    """)

if __name__ == "__main__":
    main()