# LTX-2 Video Generation Worker for RunPod Serverless
# Optimized for H100 (80GB) - fastest generation

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git-lfs \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Create app directory
WORKDIR /app

# Install RunPod SDK
RUN pip install runpod

# Install torchvision compatible with torch 2.10+ (cu128)
RUN pip install --upgrade torchvision --index-url https://download.pytorch.org/whl/cu128

# Clone and install LTX-2 packages
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /tmp/ltx2 && \
    pip install /tmp/ltx2/packages/ltx-core && \
    pip install /tmp/ltx2/packages/ltx-pipelines && \
    rm -rf /tmp/ltx2

# Install additional dependencies
RUN pip install \
    imageio \
    imageio-ffmpeg \
    accelerate \
    safetensors \
    huggingface_hub \
    fastapi \
    uvicorn \
    scipy

# Copy handler, server, docs, and test input
COPY handler.py /app/handler.py
COPY server.py /app/server.py
COPY llms.txt /app/llms.txt
COPY test_input.json /app/test_input.json

# Default to HTTP server for remote access
# Use: docker run ... python /app/handler.py  for RunPod serverless mode
CMD ["python", "-u", "/app/server.py"]
