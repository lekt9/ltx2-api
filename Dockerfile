# LTX-2 Image-to-Video API Docker Image
# Optimized for 9:16 1080p (1080x1920) video generation

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Install UV for faster dependency management
RUN pip install --upgrade pip && pip install uv

# Clone LTX-2 repository
RUN git clone https://github.com/Lightricks/LTX-2.git /app/LTX-2

# Install PyTorch with CUDA support first
RUN pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install LTX-2 packages
WORKDIR /app/LTX-2
RUN pip install -e packages/ltx-core && \
    pip install -e packages/ltx-pipelines

# Install xformers for memory optimization
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu124

# Install API dependencies
RUN pip install \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    aiofiles \
    httpx \
    huggingface_hub \
    pydantic \
    pydantic-settings

# Create directories for models and outputs
RUN mkdir -p /models /outputs /tmp/uploads

# Copy application code
WORKDIR /app
COPY app/ /app/
COPY scripts/ /scripts/

# Make scripts executable
RUN chmod +x /scripts/*.sh 2>/dev/null || true

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
