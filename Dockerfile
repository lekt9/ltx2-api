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

# Install LTX-2 packages (use git install for latest)
RUN pip install git+https://github.com/Lightricks/LTX-2.git

# Install additional dependencies
RUN pip install \
    imageio \
    imageio-ffmpeg \
    accelerate \
    safetensors \
    huggingface_hub

# Copy handler
COPY handler.py /app/handler.py

# Set the entrypoint
CMD ["python", "-u", "/app/handler.py"]
