"""
RunPod Serverless Handler for LTX-2 Video Generation

Simple API for generating videos with LTX-2.

Input format:
{
    "input": {
        "prompt": "A cat playing piano",
        "negative_prompt": "blurry, low quality",  # optional
        "width": 768,            # optional, default 768
        "height": 512,           # optional, default 512
        "num_frames": 97,        # optional, default 97 (~4 seconds at 24fps)
        "fps": 24,               # optional, default 24
        "guidance_scale": 7.5,   # optional, default 7.5
        "num_inference_steps": 30,  # optional, default 30
        "seed": null             # optional, random if not set
    }
}

Output format:
{
    "video": "base64_encoded_mp4_data",
    "duration_seconds": 4.04,
    "resolution": "768x512",
    "seed": 12345
}
"""

import base64
import os
import subprocess
import tempfile
import time
from typing import Optional

import runpod
import torch


# Global model cache
PIPELINE = None


def load_pipeline():
    """Load LTX-2 pipeline (cached globally)."""
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    print("Loading LTX-2 pipeline...")
    start = time.time()

    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

    PIPELINE = TI2VidOneStagePipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16,
    )
    PIPELINE.to("cuda")

    print(f"Pipeline loaded in {time.time() - start:.1f}s")
    return PIPELINE


def generate_video(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    fps: int = 24,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
) -> dict:
    """Generate a video from text prompt."""

    pipeline = load_pipeline()

    # Set seed for reproducibility
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"Generating video: {prompt[:50]}...")
    print(f"  Resolution: {width}x{height}, Frames: {num_frames}")

    start = time.time()

    # Generate video
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    generation_time = time.time() - start
    print(f"Generation completed in {generation_time:.1f}s")

    # Get video frames - output.frames is a list of PIL images or tensors
    frames = output.frames

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        output_path = os.path.join(tmpdir, "output.mp4")

        # Save frames
        for i, frame in enumerate(frames):
            if hasattr(frame, 'save'):
                # PIL Image
                frame.save(frame_pattern % i)
            else:
                # Tensor or numpy array
                from PIL import Image
                import numpy as np
                if torch.is_tensor(frame):
                    frame = frame.cpu().numpy()
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                if frame.shape[0] in [1, 3, 4]:  # CHW format
                    frame = frame.transpose(1, 2, 0)
                if frame.shape[-1] == 1:
                    frame = frame.squeeze(-1)
                Image.fromarray(frame).save(frame_pattern % i)

        # Encode to MP4 using ffmpeg
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path
        ], check=True, capture_output=True)

        # Read and encode as base64
        with open(output_path, "rb") as f:
            video_bytes = f.read()

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    duration = num_frames / fps

    return {
        "video": f"data:video/mp4;base64,{video_base64}",
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}",
        "fps": fps,
        "seed": seed,
        "generation_time_seconds": round(generation_time, 2),
    }


def handler(event: dict) -> dict:
    """
    RunPod handler function.

    Args:
        event: Dictionary with "input" key containing generation parameters

    Returns:
        Dictionary with generated video and metadata
    """
    try:
        input_data = event.get("input", {})

        # Validate required fields
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        # Extract optional parameters with defaults
        result = generate_video(
            prompt=prompt,
            negative_prompt=input_data.get("negative_prompt", "blurry, low quality, distorted, glitchy, watermark"),
            width=input_data.get("width", 768),
            height=input_data.get("height", 512),
            num_frames=input_data.get("num_frames", 97),
            fps=input_data.get("fps", 24),
            guidance_scale=input_data.get("guidance_scale", 7.5),
            num_inference_steps=input_data.get("num_inference_steps", 30),
            seed=input_data.get("seed"),
        )

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Start the serverless worker
runpod.serverless.start({"handler": handler})
