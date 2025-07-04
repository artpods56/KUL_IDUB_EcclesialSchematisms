FROM python:3.11-slim

# Install system dependencies for OCR and image processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-lat \
    tesseract-ocr-pol \
    tesseract-ocr-rus \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    git-lfs \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Initialize git-lfs
RUN git lfs install

# Copy application files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make setup script executable
RUN chmod +x /app/setup_data.sh

# Create app config directory
RUN mkdir -p /app/config

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command that runs setup first, then the main application
CMD ["/bin/bash", "-c", "/app/setup_data.sh && python dataset_sync.py --loop --env_file /app/config/.env.dev"]