#!/bin/bash
set -e

echo "Starting data setup process..."

# Check if HF repo is already cloned and has images
if [ ! -d "/data/hf_dataset/.git" ] || [ ! -f "/data/images/setup_complete" ]; then
    echo "Setting up HuggingFace dataset and extracting images..."
    
    # Create directories
    mkdir -p /data/images /data/label_studio_annotations /data/ocr_cache /data/hf_dataset
    
    # Clone HF repo if not exists
    if [ ! -d "/data/hf_dataset/.git" ]; then
        echo "Cloning HuggingFace repository..."
        if [ -n "${HF_TOKEN}" ] && [ -n "${HF_REPO_URL}" ]; then
            echo "Using authenticated clone..."
            git clone "https://user:${HF_TOKEN}@huggingface.co/datasets/${HF_REPO_URL#*datasets/}" "/data/hf_dataset" || \
            git clone "https://huggingface.co/datasets/${HF_REPO_URL#*datasets/}" "/data/hf_dataset"
        elif [ -n "${HF_REPO_URL}" ]; then
            echo "Using public clone..."
            git clone "https://huggingface.co/datasets/${HF_REPO_URL#*datasets/}" "/data/hf_dataset"
        else
            echo "No HF_REPO_URL provided, skipping repository clone"
        fi
    fi
    
    # Extract images if not already done
    if [ -d "/data/hf_dataset" ] && [ ! -f "/data/images/setup_complete" ]; then
        echo "Extracting images from tar files..."
        cd "/data/hf_dataset"
        
        # Pull LFS files
        echo "Pulling git-lfs files..."
        git lfs pull || echo "Warning: git lfs pull failed, continuing with available files"
        
        # Extract various archive formats
        echo "Looking for archive files to extract..."
        find . -name '*.tar.gz' -exec bash -c 'echo "Extracting: $1"; tar -xzvf "$1" -C "/data/images"' _ {} \; || true
        find . -name '*.tar' -exec bash -c 'echo "Extracting: $1"; tar -xvf "$1" -C "/data/images"' _ {} \; || true
        find . -name '*.zip' -exec bash -c 'echo "Extracting: $1"; unzip -o "$1" -d "/data/images"' _ {} \; || true
        
        # Count extracted files
        image_count=$(find /data/images -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.tiff" -o -name "*.bmp" \) | wc -l)
        echo "Extracted $image_count image files"
        
        # Mark setup as complete
        touch "/data/images/setup_complete"
        echo "Image extraction completed successfully"
    fi
else
    echo "HuggingFace dataset already set up, skipping setup process"
fi

# Ensure all required directories exist
mkdir -p /data/images /data/label_studio_annotations /data/ocr_cache /data/hf_dataset

echo "Data setup process completed"