import os
import io
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from vllm import LLM, SamplingParams
from PIL import Image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
import uvicorn

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "ds4sd/SmolDocling-256M-preview")
OUTPUT_DIR = Path("out/")
PROMPT_TEXT = "Convert page to Docling."
MAX_TOKENS = 8192 # Default from run_docling.py

# Global variable to hold the LLM instance
llm_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global llm_instance
    print(f"Loading model: {MODEL_PATH}")
    #limit_mm_per_prompt only supported in latest vllm, adjust if needed
    try:
        # Force dtype to float16 to avoid potential BFloat16 issues
        llm_instance = LLM(model=MODEL_PATH, dtype='float16') # Use float16 instead of BFloat16
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # If the model fails to load, the API won't be useful.
        # Consider how to handle this - maybe exit or run without model?
        # For now, it will proceed but OCR endpoint will fail.
        llm_instance = None
    yield
    # Clean up the ML models and release the resources
    print("Shutting down...")
    # No specific cleanup needed for vLLM unless resources need explicit release

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(lifespan=lifespan, title="Ecclesiastical OCR API", version="0.1.0")

# --- Sampling Parameters ---
# Moved here to be potentially configurable later
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_TOKENS
)

# --- Chat Template ---
# Moved here for consistency
chat_template = f"<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance>Assistant:"


# --- API Endpoints ---

@app.get("/", summary="API Status")
async def get_status():
    """Returns the status of the API and the loaded model."""
    model_status = f"Model '{MODEL_PATH}' loaded." if llm_instance else f"Model '{MODEL_PATH}' failed to load."
    return JSONResponse(content={"status": "available", "model": MODEL_PATH, "message": model_status})

# Placeholder for /ocr/ and /outputs/ endpoints to be added next
@app.post("/ocr/", summary="Perform OCR on Uploaded Image")
async def perform_ocr(file: UploadFile = File(...)):
    """Receives an image file, processes it using SmolDocling, saves DocTags and Markdown, and returns paths."""
    start_time = time.time()

    if llm_instance is None:
        raise HTTPException(status_code=503, detail="Model is not available or failed to load.")

    try:
        # Read image data
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading or opening image file: {e}")
    finally:
        await file.close()

    try:
        # Prepare LLM input
        llm_input = {"prompt": chat_template, "multi_modal_data": {"image": image}}

        # Run inference
        api_start_time = time.time()
        output = llm_instance.generate([llm_input], sampling_params=sampling_params)[0]
        api_processing_time = time.time() - api_start_time

        doctags = output.outputs[0].text

        # Generate unique base filename
        # Use original filename + timestamp to avoid collisions
        base_filename = f"{Path(file.filename).stem}_{int(time.time())}"

        # Save DocTags
        output_filename_dt = f"{base_filename}.dt"
        output_path_dt = OUTPUT_DIR / output_filename_dt
        with open(output_path_dt, "w", encoding="utf-8") as f:
            f.write(doctags)

        # Convert to DoclingDocument and save as Markdown
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name=base_filename)
        doc.load_from_doctags(doctags_doc)

        output_filename_md = f"{base_filename}.md"
        output_path_md = OUTPUT_DIR / output_filename_md
        doc.save_as_markdown(output_path_md)

        total_processing_time = time.time() - start_time

        return JSONResponse(content={
            "message": "OCR processing successful",
            "original_filename": file.filename,
            "doctags_path": str(output_path_dt), # Return relative path from API root
            "markdown_path": str(output_path_md),
            "processing_time": api_processing_time, # Time spent in vLLM
            "total_request_time": total_processing_time # Full time for the request
        })

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during OCR processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during OCR processing: {e}")

@app.get("/outputs/{filename}", summary="Get Output File")
async def get_output_file(filename: str):
    """Serves a file from the output directory."""
    # Basic security check to prevent path traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = OUTPUT_DIR / filename

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Determine media type based on extension (optional but helpful)
    media_type = None
    if filename.endswith(".md"):
        media_type = "text/markdown"
    elif filename.endswith(".dt"):
        media_type = "text/plain" # Or a custom type if preferred
    elif filename.endswith(".html"):
        media_type = "text/html"

    return FileResponse(path=file_path, media_type=media_type, filename=filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 