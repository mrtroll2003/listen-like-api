#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo ">>> Installing build dependencies and ffmpeg..."
# Update package list
apt-get update

# Install build tools for sentencepiece and ffmpeg itself
# Using -yq for non-interactive and quiet installation
apt-get install -yq --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg

echo ">>> Dependencies installed."

# --- Copy ffmpeg binary to the deployment directory ---
# Vercel typically copies the contents of the 'api' directory for the function.
# We'll copy ffmpeg into the 'api' directory so it's included.
# Adjust the destination if your function file isn't directly in api/
echo ">>> Copying ffmpeg binary to api/ directory..."
cp /usr/bin/ffmpeg api/ffmpeg
# Make sure the copied binary is executable (permissions might get lost)
chmod +x api/ffmpeg
echo ">>> ffmpeg copied and made executable in api/."

# --- Install Python dependencies ---
echo ">>> Installing Python dependencies from requirements.txt..."
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
echo ">>> Python dependencies installed."

echo ">>> Build script finished."