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
    generate_controlnet_video,
    generate_keyframe_video,
    generate_video_extension,
    generate_audio_for_video,
    video_to_base64,
    init_pipeline,
    MODEL_VARIANT,
    MODEL_REPO,
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    _decode_base64_image,
    _decode_base64_video,
    _decode_base64_mask,
    _decode_keyframes,
    _resolve_lora_entries,
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
    image_start: Optional[str] = Field(default=None, description="Base64-encoded image for first frame conditioning")
    image_end: Optional[str] = Field(default=None, description="Base64-encoded image for last frame conditioning")
    image_strength: float = Field(default=1.0, description="Image conditioning strength (0.0-1.0)")

    # Generation mode selector
    mode: str = Field(
        default="text_to_video",
        description="Generation mode: text_to_video, image_to_video, controlnet, keyframe_interpolation, video_extension, video_to_audio, inpainting"
    )

    # ControlNet (IC-LoRA)
    control_type: Optional[str] = Field(default=None, description="Control type: depth, pose, canny")
    control_video: Optional[str] = Field(default=None, description="Base64-encoded control signal video (MP4)")
    control_strength: float = Field(default=1.0, description="Control signal strength (0.0-1.0)")

    # Keyframe Interpolation
    keyframes: Optional[list] = Field(default=None, description="List of keyframe dicts: [{image: base64, frame_index: int, strength: float}]")

    # Video Extension / Video-to-Audio / Inpainting (shared input)
    source_video: Optional[str] = Field(default=None, description="Base64-encoded source video (MP4)")
    num_continuation_frames: Optional[int] = Field(default=None, description="Number of frames to generate for video extension")

    # Masking/Inpainting
    mask: Optional[str] = Field(default=None, description="Base64-encoded mask image (white=generate, black=preserve)")
    mask_strength: float = Field(default=1.0, description="Mask strength (0.0-1.0)")
    mask_start_frame: int = Field(default=0, description="Frame index where mask starts")

    # LoRA
    loras: Optional[list] = Field(default=None, description="List of LoRA dicts: [{path: str, strength: float}]")

    # FP8 quantization mode
    fp8: bool = Field(default=False, description="Use FP8 quantization for lower VRAM usage")


class VideoResponse(BaseModel):
    video: str
    fps: float
    height: int
    width: int
    num_frames: int
    seed: int
    generation_time_seconds: float
    has_audio: bool
    mode: str = "text_to_video"
    has_image_start: bool = False
    has_image_end: bool = False


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
            "audio_input": "Condition video generation on audio input",
            "two_stage": "Two-stage generation with spatial upsampling (2x)",
            "controlnet": "IC-LoRA control via depth, pose, or canny signals",
            "keyframe_interpolation": "Interpolate between keyframe images",
            "video_extension": "Extend existing videos with continuation frames",
            "video_to_audio": "Generate audio for existing video (foley)",
            "inpainting": "Mask-based video inpainting",
            "lora": "Custom LoRA weight loading",
            "fp8": "FP8 quantization for lower VRAM usage",
        },
        "endpoints": {
            "POST /generate": "Generate video (mode-based routing)",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation",
        },
        "modes": {
            "text_to_video": "Text prompt only",
            "image_to_video": "Text prompt + image_start/image_end",
            "controlnet": "Text + control_type + control_video",
            "keyframe_interpolation": "Text + keyframes list",
            "video_extension": "Text + source_video + num_continuation_frames",
            "video_to_audio": "Text + source_video (generates audio track)",
            "inpainting": "Text + source_video + mask",
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
    Generate a video with synchronized audio.

    Use the `mode` field to select the generation type:
    - **text_to_video** (default): Generate from text prompt
    - **image_to_video**: Generate from text + start/end images
    - **controlnet**: Generate with IC-LoRA control signals (depth/pose/canny)
    - **keyframe_interpolation**: Interpolate between keyframe images
    - **video_extension**: Extend an existing video with new frames
    - **video_to_audio**: Generate audio for an existing video
    - **inpainting**: Mask-based video inpainting

    All modes support optional LoRA weights and FP8 quantization.
    """
    def run_generation():
        with generation_lock:
            mode = request.mode
            temp_files = []

            try:
                # Decode audio if provided
                audio_data = None
                if request.audio:
                    audio_base64 = request.audio
                    if audio_base64.startswith("data:"):
                        audio_base64 = audio_base64.split(",", 1)[1]
                    audio_data = base64.b64decode(audio_base64)

                # Decode images if provided
                image_start = _decode_base64_image(request.image_start)
                image_end = _decode_base64_image(request.image_end)

                # Resolve LoRA entries
                lora_entries = _resolve_lora_entries(request.loras)

                # Decode source video if provided (used by multiple modes)
                source_video_path = None
                if request.source_video:
                    source_video_path = _decode_base64_video(request.source_video)
                    if source_video_path:
                        temp_files.append(source_video_path)

                # Decode control video if provided
                control_video_path = None
                if request.control_video:
                    control_video_path = _decode_base64_video(request.control_video)
                    if control_video_path:
                        temp_files.append(control_video_path)

                # Route based on mode
                if mode == "controlnet":
                    if not control_video_path:
                        raise ValueError("control_video is required for controlnet mode")
                    if not request.control_type:
                        raise ValueError("control_type is required for controlnet mode")
                    result = generate_controlnet_video(
                        prompt=request.prompt,
                        control_type=request.control_type,
                        control_video_path=control_video_path,
                        control_strength=request.control_strength,
                        negative_prompt=request.negative_prompt,
                        num_frames=request.num_frames,
                        height=request.height,
                        width=request.width,
                        fps=request.fps,
                        seed=request.seed or 0,
                        tile_size=request.tile_size,
                        image_start=image_start,
                        image_end=image_end,
                        image_strength=request.image_strength,
                        audio_data=audio_data,
                        audio_strength=request.audio_strength,
                        lora_entries=lora_entries,
                        fp8=request.fp8,
                    )

                elif mode == "keyframe_interpolation":
                    if not request.keyframes:
                        raise ValueError("keyframes list is required for keyframe_interpolation mode")
                    keyframes = _decode_keyframes(request.keyframes)
                    if not keyframes:
                        raise ValueError("No valid keyframes provided")
                    result = generate_keyframe_video(
                        prompt=request.prompt,
                        keyframes=keyframes,
                        negative_prompt=request.negative_prompt,
                        num_frames=request.num_frames,
                        height=request.height,
                        width=request.width,
                        fps=request.fps,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        seed=request.seed or 0,
                        tile_size=request.tile_size,
                        audio_data=audio_data,
                        audio_strength=request.audio_strength,
                        lora_entries=lora_entries,
                        fp8=request.fp8,
                    )

                elif mode == "video_extension":
                    if not source_video_path:
                        raise ValueError("source_video is required for video_extension mode")
                    result = generate_video_extension(
                        prompt=request.prompt,
                        source_video_path=source_video_path,
                        num_continuation_frames=request.num_continuation_frames or 61,
                        negative_prompt=request.negative_prompt,
                        height=request.height,
                        width=request.width,
                        fps=request.fps,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        seed=request.seed or 0,
                        tile_size=request.tile_size,
                        audio_data=audio_data,
                        audio_strength=request.audio_strength,
                        lora_entries=lora_entries,
                        fp8=request.fp8,
                    )

                elif mode == "video_to_audio":
                    if not source_video_path:
                        raise ValueError("source_video is required for video_to_audio mode")
                    result = generate_audio_for_video(
                        prompt=request.prompt,
                        source_video_path=source_video_path,
                        negative_prompt=request.negative_prompt,
                        fps=request.fps,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        seed=request.seed or 0,
                        tile_size=request.tile_size,
                        lora_entries=lora_entries,
                        fp8=request.fp8,
                    )

                elif mode == "inpainting":
                    if not source_video_path:
                        raise ValueError("source_video is required for inpainting mode")
                    mask_tensor = _decode_base64_mask(request.mask, request.num_frames)
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
                        image_start=image_start,
                        image_end=image_end,
                        image_strength=request.image_strength,
                        source_video_path=source_video_path,
                        mask_tensor=mask_tensor,
                        mask_strength=request.mask_strength,
                        mask_start_frame=request.mask_start_frame,
                        lora_entries=lora_entries,
                        fp8=request.fp8,
                    )

                else:
                    # text_to_video / image_to_video (default)
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
                        image_start=image_start,
                        image_end=image_end,
                        image_strength=request.image_strength,
                        lora_entries=lora_entries,
                        fp8=request.fp8,
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
                    "mode": mode,
                    "has_image_start": image_start is not None,
                    "has_image_end": image_end is not None,
                }

            finally:
                # Clean up temp files
                for tmp_path in temp_files:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

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
