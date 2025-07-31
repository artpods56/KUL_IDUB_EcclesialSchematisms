#!/bin/sh
set -e

# Start ollama serve in the background
ollama serve &
pid=$!

# Wait a bit for the server to start
echo "Waiting for Ollama server to start..."
sleep 5
echo "Ollama server should be ready."

# Pull the model
echo "Pulling model gemma2:27b..."
ollama pull gemma2:27b
echo "Model pulled."

# Keep the container running by waiting for the ollama serve process
echo "Ollama is running..."
wait $pid
