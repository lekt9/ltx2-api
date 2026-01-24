"""
HTTP Server wrapper for LTX-2 Video Generation

Provides a simple HTTP API for remote access to the video generation handler.
Following Wan2GP implementation pattern with DeepBeepMeep/LTX-2 models.
Generates synchronized audio-video content.
"""

import asyncio
import base64
import os
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Import the handler functions
from handler import (
    generate_video,
    video_to_base64,
    init_pipeline,
    MODEL_VARIANT,
    MODEL_REPO,
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
)

# Lock to ensure only one generation at a time per worker
generation_lock = Lock()

app = FastAPI(
    title="LTX-2 Video Generation API",
    description="Generate AI videos with synchronized audio using the LTX-2 19B model (DeepBeepMeep variant). Supports text-to-video and audio-to-video.",
    version="2.0.0",
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
    width: int = Field(default=768, description="Video width (must be divisible by 64)")
    height: int = Field(default=512, description="Video height (must be divisible by 64)")
    num_frames: int = Field(default=121, description="Number of frames (~5 seconds at 24fps)")
    fps: float = Field(default=DEFAULT_FPS, description="Frames per second")
    guidance_scale: float = Field(default=DEFAULT_GUIDANCE_SCALE, description="CFG guidance scale (recommended: 4.0)")
    num_inference_steps: int = Field(default=DEFAULT_NUM_INFERENCE_STEPS, description="Quality steps (default: 40)")
    seed: Optional[int] = Field(default=None, description="Seed for reproducibility")
    include_audio: bool = Field(default=True, description="Include generated audio in output")
    audio: Optional[str] = Field(default=None, description="Base64-encoded WAV audio for audio-to-video conditioning")
    audio_strength: float = Field(default=1.0, description="Audio conditioning strength (0.0-1.0)")
    tile_size: Optional[int] = Field(default=None, description="VAE tiling size for memory optimization")


class VideoResponse(BaseModel):
    video: str
    fps: float
    height: int
    width: int
    num_frames: int
    seed: int
    generation_time_seconds: float
    has_audio: bool


@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup."""
    print("Pre-loading LTX-2 pipeline...")
    try:
        init_pipeline(MODEL_VARIANT)
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
        "model": f"{MODEL_REPO} ({MODEL_VARIANT} variant)",
        "features": {
            "audio_output": "Generates synchronized audio with video",
            "audio_input": "SUPPORTED - condition video generation on audio input",
            "two_stage": "Two-stage generation with spatial upsampling (2x)",
        },
        "endpoints": {
            "POST /generate": "Generate video with audio from text",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
        },
        "modes": {
            "text-to-video": "Provide only 'prompt' parameter",
            "audio-to-video": "Provide 'prompt' and 'audio' (base64 WAV) parameters",
        },
        "defaults": {
            "resolution": "768x512",
            "frames": 121,
            "fps": DEFAULT_FPS,
            "duration": "~5 seconds",
            "audio": "included (24kHz)",
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
        },
        "note": "Dimensions should be divisible by 64 for optimal results",
    }


@app.get("/busy")
async def busy():
    """Check if worker is currently processing a request."""
    return {"busy": generation_lock.locked()}


@app.post("/generate", response_model=VideoResponse)
async def generate(request: VideoRequest):
    """
    Generate a video with synchronized audio from text prompt.

    Dimensions should be divisible by 64 for optimal results.

    For audio-to-video, provide a base64-encoded WAV audio in the 'audio' field.

    Audio is generated automatically and included in the output MP4.
    Set include_audio=false to get video-only output.
    """
    def run_generation():
        with generation_lock:
            # Decode audio if provided
            audio_data = None
            if request.audio:
                audio_base64 = request.audio
                if audio_base64.startswith("data:"):
                    audio_base64 = audio_base64.split(",", 1)[1]
                audio_data = base64.b64decode(audio_base64)

            # Generate video
            result = generate_video(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_frames=request.num_frames,
                height=request.height,
                width=request.width,
                fps=request.fps,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=request.seed or 0,
                audio_data=audio_data,
                audio_strength=request.audio_strength,
                tile_size=request.tile_size,
            )

            if "error" in result:
                raise Exception(result["error"])

            # Encode video to base64
            video_base64 = video_to_base64(
                result["video"],
                fps=result["fps"],
                audio_np=result.get("audio") if request.include_audio else None,
                audio_sr=result.get("audio_sample_rate", 24000),
            )

            return {
                "video": f"data:video/mp4;base64,{video_base64}",
                "fps": result["fps"],
                "height": result["height"],
                "width": result["width"],
                "num_frames": result["num_frames"],
                "seed": result["seed"],
                "generation_time_seconds": round(result["generation_time"], 2),
                "has_audio": request.include_audio and result.get("audio") is not None,
            }

    try:
        # Run in thread pool to not block the event loop
        result = await asyncio.get_event_loop().run_in_executor(None, run_generation)
        return result
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Generation error: {error_msg}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
