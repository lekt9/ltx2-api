#!/bin/bash
# LTX-2 API Setup Script

set -e

echo "========================================"
echo "LTX-2 Image-to-Video API Setup"
echo "========================================"

# Check for NVIDIA Docker runtime
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: NVIDIA Docker runtime not detected."
    echo "Please install nvidia-container-toolkit:"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
fi

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable not set."
    echo "You'll need it to download Gemma model."
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Enter your HuggingFace token (or press Enter to skip): " HF_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        export HF_TOKEN
    fi
fi

# Create directories
echo "Creating directories..."
mkdir -p models outputs tmp

# Download models
echo ""
echo "Downloading models (this may take a while)..."
echo "Required disk space: ~50GB"
echo ""

if [ -n "$HF_TOKEN" ]; then
    docker-compose --profile download run model-downloader
else
    echo "Skipping model download (no HF_TOKEN)."
    echo "Please run: HF_TOKEN=your_token docker-compose --profile download run model-downloader"
fi

# Build the API image
echo ""
echo "Building Docker image..."
docker-compose build

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To start the API server:"
echo "  docker-compose up -d"
echo ""
echo "To check logs:"
echo "  docker-compose logs -f"
echo ""
echo "API will be available at:"
echo "  http://localhost:8000"
echo ""
echo "Health check:"
echo "  curl http://localhost:8000/health"
echo ""
