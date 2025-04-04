import streamlit as st
import os
from pathlib import Path
from PIL import Image
import io
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, stream_generate

# Load model (do it once at startup)
@st.cache_resource
def load_model():
    model_path = "ds4sd/SmolDocling-256M-preview-mlx-bf16"
    model, processor = load(model_path)
    config = load_config(model_path)
    return model, processor, config

# Title and description
st.set_page_config(page_title="SmolDocling OCR", layout="wide")
st.title("Ecclesiastical OCR with SmolDocling")
st.markdown("""
Upload multiple images (in the correct order) to perform OCR and merge them into a single document.
""")

# Initialize model
model, processor, config = load_model()

# Support functions for inference
def process_single_image(image, prompt="Convert this page to docling."):
    # Apply chat template
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
    
    # Generate output
    output = ""
    for token in stream_generate(
        model, processor, formatted_prompt, [image], max_tokens=4096, verbose=False
    ):
        output += token.text
        if "</doctag>" in token.text:
            break
            
    return output

def process_batch(images, prompt):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, img in enumerate(images):
        status_text.text(f"Processing image {i+1}/{len(images)}...")
        results.append(process_single_image(img, prompt))
        progress_bar.progress((i+1)/len(images))
    
    status_text.text("Creating document...")
    
    # Populate document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(results, images)
    # Create a docling document
    doc = DoclingDocument(name="BatchDocument")
    doc.load_from_doctags(doctags_doc)
    
    status_text.text("Processing complete!")
    return doc

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    prompt = st.text_area("Prompt", value="Convert this page to docling.", height=100)
    output_format = st.selectbox(
        "Output Format",
        ["HTML", "Markdown"],
        help="Select the desired output format"
    )
    
    st.header("Additional Options")
    image_mode = st.selectbox(
        "Image Mode for HTML", 
        ["EMBEDDED", "REFERENCED"],
        help="How images are stored in HTML output"
    )
    
    img_ref_mode = ImageRefMode.EMBEDDED if image_mode == "EMBEDDED" else ImageRefMode.REFERENCED

# Main interface
uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} images")
    
    # Show thumbnails with numbers to indicate order
    cols = st.columns(4)
    for i, file in enumerate(uploaded_files):
        with cols[i % 4]:
            st.image(file, caption=f"Image {i+1}", width=150)
    
    # Process button
    if st.button("Process Images"):
        with st.spinner("Loading images..."):
            # Load all images
            images = [Image.open(file) for file in uploaded_files]
        
        # Process images and show result
        doc = process_batch(images, prompt)
        
        # Display based on selected format
        if output_format == "HTML":
            output_path = Path("./batch_output.html")
            doc.save_as_html(output_path, image_mode=img_ref_mode)
            
            with open(output_path, "r") as f:
                html_content = f.read()
            
            st.download_button(
                label="Download HTML",
                data=html_content,
                file_name="document.html",
                mime="text/html"
            )
            
            # Display preview
            st.subheader("HTML Preview")
            st.components.v1.html(html_content, height=600, scrolling=True)
            
        elif output_format == "Markdown":
            markdown_content = doc.export_to_markdown()
            
            st.download_button(
                label="Download Markdown",
                data=markdown_content,
                file_name="document.md",
                mime="text/markdown"
            )
            
            # Display preview
            st.subheader("Markdown Preview")
            st.text(markdown_content)
else:
    st.info("Please upload images to begin processing")