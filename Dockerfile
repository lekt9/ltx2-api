# LTX-2 Video Generation Worker for RunPod Serverless
# B200 (Blackwell/sm_100) compatible with PyTorch 2.9.x + CUDA 12.8

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Build arg for GPU architectures (sm_100 = Blackwell/B200)
ARG CUDA_ARCHITECTURES="10.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch
ENV MODEL_VARIANT=distilled

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl \
    git-lfs ffmpeg libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Install PyTorch 2.9.x with CUDA 12.8 and sm_100 (Blackwell) support
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
    'torch>=2.9.0,<2.10.0' 'torchvision>=0.24.0,<0.25.0' 'torchaudio>=2.9.0,<2.10.0'

# Install core dependencies
RUN pip install \
    runpod \
    accelerate \
    safetensors \
    huggingface_hub \
    transformers \
    einops \
    tqdm \
    scipy \
    imageio \
    imageio-ffmpeg \
    fastapi \
    uvicorn \
    Pillow \
    sentencepiece \
    av

# Copy local ltx_core and ltx_pipelines (from Wan2GP)
COPY ltx_core /app/ltx_core
COPY ltx_pipelines /app/ltx_pipelines
COPY shared /app/shared

# Copy handler, server, docs, and test input
COPY handler.py /app/handler.py
COPY server.py /app/server.py
COPY llms.txt /app/llms.txt
COPY test_input.json /app/test_input.json

# Default to HTTP server for remote access
CMD ["python3", "-u", "/app/server.py"]
