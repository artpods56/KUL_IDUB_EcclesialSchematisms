# Replace the existing Dockerfile with an updated version that uses uv for dependency installation and runs the evaluation workflow by default.
FROM python:3.12-slim-bookworm

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System dependencies required for Tesseract OCR and common scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Create working directory
WORKDIR /app

# ---------------------------------------------------------------------------
# Copy ONLY dependency files first for caching
# ---------------------------------------------------------------------------
COPY pyproject.toml .

# Copy src/ (required for editable install)
COPY src/ src/

# Install with uv, using a cache mount to persist downloads across builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e /app

# ---------------------------------------------------------------------------
# Now copy the rest of the project (configs, data, etc.)
COPY . .

# Create cache directory
RUN mkdir -p /app/tmp/caches

# Set a fixed import path; runtime overrides via `-e PYTHONPATH=` still work
ENV PYTHONPATH="/app/src"

# ---------------------------------------------------------------------------
# 4. Default command – run the evaluation workflow
#    Can be overridden with `docker run ... <other command>` if desired.
# ---------------------------------------------------------------------------
CMD ["python", "src/scripts/evaluate.py"]