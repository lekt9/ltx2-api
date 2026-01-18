# LTX-2 Video Generation Worker for RunPod Serverless
# Optimized for H100 (80GB) - fastest generation

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create app directory
WORKDIR /app

# Install PyTorch with CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install RunPod SDK
RUN pip install runpod

# Install LTX-2 packages
RUN pip install ltx-core ltx-pipelines

# Install additional dependencies
RUN pip install \
    imageio \
    imageio-ffmpeg \
    accelerate \
    transformers \
    safetensors \
    huggingface_hub

# Copy handler
COPY handler.py /app/handler.py

# Pre-download model weights during build (optional - makes cold starts faster)
# Uncomment if you want models baked into the image (larger image, faster cold start)
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Lightricks/LTX-2')"

# Set the entrypoint
CMD ["python", "-u", "/app/handler.py"]
