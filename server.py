"""
HTTP Server wrapper for LTX-2 Video Generation

Provides a simple HTTP API for remote access to the video generation handler.
Generates synchronized audio-video content.
"""

import asyncio
import os
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Import the handler function
from handler import generate_video, load_pipeline

# Lock to ensure only one generation at a time per worker
generation_lock = Lock()

app = FastAPI(
    title="LTX-2 Video Generation API",
    description="Generate AI videos with synchronized audio using the LTX-2 19B model. Supports text-to-video and image-to-video.",
    version="1.0.0",
)

# Enable CORS for remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoRequest(BaseModel):
    prompt: str = Field(..., description="Text description of the video to generate")
    negative_prompt: str = Field(
        default="blurry, low quality, distorted, glitchy, watermark",
        description="What to avoid in the generation"
    )
    width: int = Field(default=768, description="Video width (must be divisible by 32)")
    height: int = Field(default=512, description="Video height (must be divisible by 32)")
    num_frames: int = Field(default=121, description="Number of frames (~5 seconds at 25fps)")
    fps: int = Field(default=25, description="Frames per second")
    guidance_scale: float = Field(default=3.0, description="CFG guidance scale (1-5)")
    num_inference_steps: int = Field(default=40, description="Quality steps (30-50)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducibility")
    image: Optional[str] = Field(default=None, description="Base64-encoded image for image-to-video mode")
    image_cond_noise_scale: float = Field(default=0.15, description="Image conditioning strength (0.0-1.0)")
    include_audio: bool = Field(default=True, description="Include generated audio in output")
    audio: Optional[str] = Field(default=None, description="Base64-encoded WAV audio for audio-to-video conditioning")
    audio_cond_noise_scale: float = Field(default=0.15, description="Audio conditioning strength (0.0-1.0)")


class VideoResponse(BaseModel):
    video: str
    duration_seconds: float
    resolution: str
    fps: int
    seed: int
    generation_time_seconds: float
    mode: str
    has_audio: bool


@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup."""
    print("Pre-loading LTX-2 pipeline...")
    try:
        load_pipeline()
        print("Pipeline loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to pre-load pipeline: {e}")
        print("Pipeline will be loaded on first request.")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "LTX-2 Video Generation API",
        "model": "Lightricks/LTX-2 (19B parameters)",
        "features": {
            "audio_output": "Generates synchronized audio with video",
            "audio_input": "SUPPORTED - condition video generation on audio input",
        },
        "endpoints": {
            "POST /generate": "Generate video with audio from text/image",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
        },
        "modes": {
            "text-to-video": "Provide only 'prompt' parameter",
            "image-to-video": "Provide 'prompt' and 'image' (base64) parameters",
            "audio-to-video": "Provide 'prompt' and 'audio' (base64 WAV) parameters",
        },
        "defaults": {
            "resolution": "768x512",
            "frames": 121,
            "fps": 25,
            "duration": "~5 seconds",
            "audio": "included (24kHz)",
        },
        "note": "For 9:16 portrait, use width=288, height=512",
    }


@app.get("/busy")
async def busy():
    """Check if worker is currently processing a request."""
    return {"busy": generation_lock.locked()}


@app.post("/generate", response_model=VideoResponse)
async def generate(request: VideoRequest):
    """
    Generate a video with synchronized audio from text prompt.

    For 9:16 portrait mode at 480p, use:
    - width: 288 (or 256)
    - height: 512 (or 448)

    For image-to-video, provide a base64-encoded image in the 'image' field.
    For audio-to-video, provide a base64-encoded WAV audio in the 'audio' field.

    Audio is generated automatically and included in the output MP4.
    Set include_audio=false to get video-only output.
    """
    def run_generation():
        with generation_lock:
            return generate_video(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                num_frames=request.num_frames,
                fps=request.fps,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                seed=request.seed,
                image=request.image,
                image_cond_noise_scale=request.image_cond_noise_scale,
                include_audio=request.include_audio,
                audio=request.audio,
                audio_cond_noise_scale=request.audio_cond_noise_scale,
            )

    try:
        # Run in thread pool to not block the event loop
        result = await asyncio.get_event_loop().run_in_executor(None, run_generation)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
