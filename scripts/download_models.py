#!/usr/bin/env python3
"""
Download LTX-2 models from HuggingFace Hub.

Models required:
- LTX-2 checkpoint (distilled FP8 recommended for lower VRAM)
- Spatial upsampler
- Distilled LoRA
- Gemma-3 text encoder

Usage:
    python download_models.py [--models-dir /path/to/models] [--full-precision]
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub")
    from huggingface_hub import hf_hub_download, snapshot_download


# LTX-2 model repository
LTX2_REPO = "Lightricks/LTX-2"

# Available checkpoints
CHECKPOINTS = {
    "distilled-fp8": "ltx-2-19b-distilled-fp8.safetensors",  # Smallest, recommended
    "distilled": "ltx-2-19b-distilled.safetensors",
    "dev-fp8": "ltx-2-19b-dev-fp8.safetensors",
    "dev": "ltx-2-19b-dev.safetensors",  # Full precision, largest
}

# Required additional models
ADDITIONAL_MODELS = [
    "ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "ltx-2-19b-distilled-lora-384.safetensors",
]

# Gemma-3 text encoder
GEMMA_REPO = "google/gemma-3-4b-it"


def download_ltx2_models(models_dir: Path, checkpoint_type: str = "distilled-fp8"):
    """Download LTX-2 models."""
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("LTX-2 Model Downloader")
    print(f"{'='*60}")
    print(f"Models directory: {models_dir}")
    print(f"Checkpoint type: {checkpoint_type}")
    print()

    # Download main checkpoint
    checkpoint_name = CHECKPOINTS.get(checkpoint_type, CHECKPOINTS["distilled-fp8"])
    print(f"[1/3] Downloading LTX-2 checkpoint: {checkpoint_name}")

    try:
        checkpoint_path = hf_hub_download(
            repo_id=LTX2_REPO,
            filename=checkpoint_name,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
        print(f"      ✓ Downloaded to: {checkpoint_path}")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        sys.exit(1)

    # Download additional models
    print(f"\n[2/3] Downloading additional LTX-2 models...")
    for model_name in ADDITIONAL_MODELS:
        print(f"      - {model_name}")
        try:
            model_path = hf_hub_download(
                repo_id=LTX2_REPO,
                filename=model_name,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
            )
            print(f"        ✓ Downloaded")
        except Exception as e:
            print(f"        ✗ Failed: {e}")
            # Continue with other models

    # Download Gemma-3 text encoder
    gemma_dir = models_dir / "gemma-3-4b-it"
    print(f"\n[3/3] Downloading Gemma-3 text encoder...")
    print(f"      Repository: {GEMMA_REPO}")

    try:
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=gemma_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.gguf", "*.bin"],  # Skip GGUF and old bin formats
        )
        print(f"      ✓ Downloaded to: {gemma_dir}")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        print("      Note: You may need to accept the license agreement at:")
        print(f"      https://huggingface.co/{GEMMA_REPO}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")
    print("\nModels downloaded:")
    for f in models_dir.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        elif f.is_dir():
            print(f"  - {f.name}/ (directory)")

    print("\nYou can now start the API server with:")
    print("  docker-compose up -d")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download LTX-2 models")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/models"),
        help="Directory to save models (default: /models)",
    )
    parser.add_argument(
        "--checkpoint",
        choices=list(CHECKPOINTS.keys()),
        default="distilled-fp8",
        help="Checkpoint type to download (default: distilled-fp8)",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Download full precision model instead of FP8",
    )

    args = parser.parse_args()

    checkpoint_type = args.checkpoint
    if args.full_precision:
        if "fp8" in checkpoint_type:
            checkpoint_type = checkpoint_type.replace("-fp8", "")

    download_ltx2_models(args.models_dir, checkpoint_type)


if __name__ == "__main__":
    main()
