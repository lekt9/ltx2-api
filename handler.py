"""
RunPod Serverless Handler for LTX-2 Video Generation

Simple API for generating videos with LTX-2.

Input format:
{
    "input": {
        "prompt": "A cat playing piano",
        "negative_prompt": "blurry, low quality",  # optional
        "width": 1280,           # optional, default 1280
        "height": 720,           # optional, default 720
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
    "resolution": "1280x720",
    "seed": 12345
}
"""

import base64
import os
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

    # Import here to avoid loading at module level
    from ltx_pipelines import LTXTextToVideoPipeline

    # Use FP16 for speed, or FP8 if available
    dtype = torch.float16

    # Check for FP8 support (RTX 40/50 series, H100)
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] >= 9:  # H100, etc.
            try:
                dtype = torch.float8_e4m3fn
                print("Using FP8 precision (H100 detected)")
            except:
                pass

    PIPELINE = LTXTextToVideoPipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=dtype,
    )
    PIPELINE.to("cuda")

    # Enable memory optimizations
    PIPELINE.enable_model_cpu_offload()

    print(f"Pipeline loaded in {time.time() - start:.1f}s")
    return PIPELINE


def generate_video(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    width: int = 1280,
    height: int = 720,
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

    # Get video frames
    video_frames = output.frames[0]  # Shape: (num_frames, height, width, 3)

    # Save to temporary MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name

    # Export to MP4 using imageio or similar
    try:
        import imageio
        writer = imageio.get_writer(temp_path, fps=fps, codec="libx264", quality=8)
        for frame in video_frames:
            # Convert to uint8 if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype("uint8")
            writer.append_data(frame)
        writer.close()
    except ImportError:
        # Fallback: use PIL and ffmpeg
        from PIL import Image
        import subprocess

        frame_dir = tempfile.mkdtemp()
        for i, frame in enumerate(video_frames):
            if frame.max() <= 1.0:
                frame = (frame * 255).astype("uint8")
            img = Image.fromarray(frame)
            img.save(f"{frame_dir}/frame_{i:04d}.png")

        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", f"{frame_dir}/frame_%04d.png",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            temp_path
        ], check=True, capture_output=True)

    # Read and encode as base64
    with open(temp_path, "rb") as f:
        video_bytes = f.read()

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")

    # Cleanup
    os.unlink(temp_path)

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
            width=input_data.get("width", 1280),
            height=input_data.get("height", 720),
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
