# SmolDocling OCR API & Interface

This directory contains two Docker services for OCR processing using SmolDocling:

1. **OCR API Service** - A FastAPI wrapper for vllm inference using SmolDocling
2. **Streamlit Interface** - A user-friendly web interface for the OCR service

## Prerequisites

- Docker with GPU support (for the OCR API service)
- Docker Compose

## Getting Started

### Configuration

1. The OCR API service uses these environment variables (set in docker-compose.yml):
   - `MODEL_PATH` - The SmolDocling model path
   - `IMAGE_DIR` - Directory for storing uploaded images
   - `OUTPUT_DIR` - Directory for storing OCR output files

2. The Streamlit interface uses:
   - `OCR_API_URL` - URL of the OCR API service

### Starting the Services

To start both services:

```bash
cd /path/to/ecclesiasticalOCR/api
docker-compose up -d
```

Then access the Streamlit interface at http://localhost:8501

### API Endpoints

The OCR API service provides these endpoints:

- `GET /` - Basic API information
- `POST /ocr/` - Process an image with OCR (multipart/form-data)
- `GET /outputs/{filename}` - Get a specific output file
- `GET /outputs/` - List all available output files

## Development and Testing

### Running Tests

For OCR API tests:

```bash
cd /path/to/ecclesiasticalOCR/api
pytest tests/
```

For Streamlit interface tests:

```bash
cd /path/to/ecclesiasticalOCR/api/streamlit
pytest tests/
```

### Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   # For OCR API
   pip install -r requirements.txt
   
   # For Streamlit interface
   pip install -r streamlit/requirements.txt
   ```

3. Run services locally:
   ```bash
   # For OCR API
   uvicorn app:app --reload
   
   # For Streamlit interface
   cd streamlit
   streamlit run app.py
   ```

## Project Structure

```
api/
├── app.py                # FastAPI application for OCR
├── Dockerfile            # Dockerfile for OCR API service
├── docker-compose.yml    # Docker Compose file for both services
├── requirements.txt      # Python dependencies for OCR API
├── tests/                # Tests for OCR API
│   └── test_app.py
├── streamlit/            # Streamlit interface
│   ├── app.py            # Streamlit application
│   ├── Dockerfile        # Dockerfile for Streamlit service
│   ├── requirements.txt  # Python dependencies for Streamlit
│   ├── .env.example      # Example environment variables
│   └── tests/            # Tests for Streamlit interface
│       └── test_app.py
├── img/                  # Directory for uploaded images (created by Docker)
└── out/                  # Directory for OCR output (created by Docker)
```

## License

[Your license information here]