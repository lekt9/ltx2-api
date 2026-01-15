"""
LTX-2 Image-to-Video API Server

FastAPI server for generating 9:16 vertical videos from images using the LTX-2 model.
Uses TI2VidTwoStagesPipeline for production-quality output with 2x spatial upsampling.
Dimensions must be divisible by 64 for the two-stage pipeline.
"""

import asyncio
import base64
import logging
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model paths
    checkpoint_path: str = "/models/ltx-2-19b-distilled-fp8.safetensors"
    gemma_root: str = "/models/gemma-3-12b-it"
    spatial_upsampler_path: str = "/models/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    distilled_lora_path: str = "/models/ltx-2-19b-distilled-lora-384.safetensors"
    distilled_lora_strength: float = 0.6

    # Generation defaults - 9:16 aspect ratio for vertical video
    # Note: dimensions must be divisible by 64 for two-stage pipeline
    # 512x896 base + 2x upscale = 1024x1792 final output (fits in 80GB VRAM)
    default_width: int = 512  # 512/64 = 8
    default_height: int = 896  # 896/64 = 14 (maintains ~9:16 ratio)
    default_num_frames: int = 121  # ~5 seconds at 25fps
    default_frame_rate: float = 25.0
    default_num_inference_steps: int = 30
    default_cfg_guidance_scale: float = 3.0

    # Server settings
    output_dir: str = "/outputs"
    temp_dir: str = "/tmp/uploads"
    max_concurrent_jobs: int = 1
    enable_fp8: bool = True

    class Config:
        env_prefix = "LTX_"


settings = Settings()

# Global pipeline instance
pipeline = None
generation_semaphore = None


class GenerationRequest(BaseModel):
    """Request model for video generation."""

    prompt: str = Field(..., description="Text prompt describing the video to generate")
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt for generation",
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    width: int = Field(default=settings.default_width, ge=256, le=2048)
    height: int = Field(default=settings.default_height, ge=256, le=2048)
    num_frames: int = Field(default=settings.default_num_frames, ge=9, le=257)
    frame_rate: float = Field(default=settings.default_frame_rate, ge=1.0, le=60.0)
    num_inference_steps: int = Field(default=settings.default_num_inference_steps, ge=1, le=100)
    cfg_guidance_scale: float = Field(default=settings.default_cfg_guidance_scale, ge=1.0, le=20.0)
    enhance_prompt: bool = Field(default=True, description="Use prompt enhancement")
    image_strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Image conditioning strength")
    image_frame: int = Field(default=0, ge=0, description="Frame index for image conditioning")


class GenerationResponse(BaseModel):
    """Response model for video generation."""

    job_id: str
    status: str
    message: str
    video_url: Optional[str] = None
    created_at: str


class JobStatus(BaseModel):
    """Job status model."""

    job_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


# In-memory job tracking
jobs: dict[str, JobStatus] = {}


def load_pipeline():
    """Load the LTX-2 pipeline."""
    global pipeline

    logger.info("Loading LTX-2 pipeline...")

    try:
        # Check if all model files exist
        required_files = [
            settings.checkpoint_path,
            settings.spatial_upsampler_path,
            settings.distilled_lora_path,
        ]

        for f in required_files:
            if not Path(f).exists():
                raise FileNotFoundError(f"Required model file not found: {f}")

        if not Path(settings.gemma_root).exists():
            raise FileNotFoundError(f"Gemma model directory not found: {settings.gemma_root}")

        # Use TI2VidTwoStagesPipeline for production-quality output
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        # Configure distilled LoRA for stage 2 refinement
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                settings.distilled_lora_path,
                settings.distilled_lora_strength,
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]

        logger.info("Using TI2VidTwoStagesPipeline (production quality with 2x upsampling)")
        logger.info(f"  Checkpoint: {settings.checkpoint_path}")
        logger.info(f"  Spatial upsampler: {settings.spatial_upsampler_path}")
        logger.info(f"  Distilled LoRA: {settings.distilled_lora_path} (strength: {settings.distilled_lora_strength})")
        logger.info(f"  Gemma: {settings.gemma_root}")
        logger.info(f"  FP8 enabled: {settings.enable_fp8}")

        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=settings.checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=settings.spatial_upsampler_path,
            gemma_root=settings.gemma_root,
            loras=[],
            fp8transformer=settings.enable_fp8,
        )

        logger.info("LTX-2 pipeline loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global generation_semaphore

    # Startup
    logger.info("Starting LTX-2 API server...")

    # Create output directories
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.temp_dir).mkdir(parents=True, exist_ok=True)

    # Initialize semaphore for concurrent job limiting
    generation_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available! Running on CPU will be very slow.")

    # Try to load pipeline (skip if models not present for health checks)
    try:
        load_pipeline()
    except FileNotFoundError as e:
        logger.warning(f"Models not found, pipeline not loaded: {e}")
        logger.info("Download models and restart the server to enable generation.")

    yield

    # Shutdown
    logger.info("Shutting down LTX-2 API server...")


app = FastAPI(
    title="LTX-2 Image-to-Video API",
    description="Generate 9:16 vertical videos from images using LTX-2",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    cuda_available = torch.cuda.is_available()
    pipeline_loaded = pipeline is not None

    return {
        "status": "healthy" if cuda_available else "degraded",
        "cuda_available": cuda_available,
        "pipeline_loaded": pipeline_loaded,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            if cuda_available
            else None
        ),
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "LTX-2 Image-to-Video API",
        "version": "1.0.0",
        "description": "Generate 9:16 vertical videos from images",
        "endpoints": {
            "POST /generate": "Generate video from image",
            "GET /jobs/{job_id}": "Check job status",
            "GET /videos/{job_id}": "Download generated video",
            "GET /health": "Health check",
        },
        "default_settings": {
            "base_resolution": f"{settings.default_width}x{settings.default_height}",
            "output_resolution": f"{settings.default_width * 2}x{settings.default_height * 2}",
            "upscale_factor": "2x",
            "aspect_ratio": "9:16",
            "num_frames": settings.default_num_frames,
            "frame_rate": settings.default_frame_rate,
            "duration_seconds": round(settings.default_num_frames / settings.default_frame_rate, 2),
        },
    }


def _run_pipeline_sync(
    image_path: str,
    output_path: str,
    request: GenerationRequest,
    seed: int,
):
    """Synchronous pipeline execution for running in executor."""
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_pipelines.utils.media_io import encode_video

    # Prepare image conditioning: (path, frame_index, strength)
    images = [(image_path, request.image_frame, request.image_strength)]

    # Configure tiling for memory efficiency
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(request.num_frames, tiling_config)

    # Run the two-stage pipeline
    # Stage 1: Generate at base resolution with CFG guidance
    # Stage 2: 2x spatial upsampling with distilled LoRA refinement
    # Returns (video_iterator, audio_tensor)
    video_iter, audio = pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        seed=seed,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        frame_rate=request.frame_rate,
        num_inference_steps=request.num_inference_steps,
        cfg_guidance_scale=request.cfg_guidance_scale,
        images=images,
        tiling_config=tiling_config,
        enhance_prompt=request.enhance_prompt,
    )

    # Encode the video to MP4
    encode_video(
        video=video_iter,
        fps=int(request.frame_rate),
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )

    return output_path


async def run_generation(
    job_id: str,
    image_path: str,
    request: GenerationRequest,
):
    """Run video generation in background."""
    global pipeline

    try:
        jobs[job_id].status = "processing"
        logger.info(f"Starting generation for job {job_id}")

        if pipeline is None:
            raise RuntimeError("Pipeline not loaded. Please download models first.")

        # Generate seed if not provided
        seed = request.seed if request.seed is not None else torch.randint(0, 2**32, (1,)).item()

        # Output path
        output_path = str(Path(settings.output_dir) / f"{job_id}.mp4")

        # Run generation
        logger.info(f"Generating video: {request.width}x{request.height}, {request.num_frames} frames")

        # Run synchronously - the pipeline handles its own threading
        # Using executor causes issues with torch.inference_mode() context
        _run_pipeline_sync(image_path, output_path, request, seed)

        # Update job status
        jobs[job_id].status = "completed"
        jobs[job_id].video_url = f"/videos/{job_id}"
        jobs[job_id].completed_at = datetime.utcnow().isoformat()
        logger.info(f"Generation completed for job {job_id}")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Generation failed for job {job_id}: {e}\n{tb}")
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
        jobs[job_id].completed_at = datetime.utcnow().isoformat()

    finally:
        # Cleanup temp image
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception:
            pass

        # Release semaphore
        generation_semaphore.release()


@app.post("/generate", response_model=GenerationResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    image: Annotated[UploadFile, File(description="Input image for video generation")],
    prompt: Annotated[str, Form(description="Text prompt describing the video")],
    negative_prompt: Annotated[
        str,
        Form(description="Negative prompt"),
    ] = "worst quality, inconsistent motion, blurry, jittery, distorted",
    seed: Annotated[Optional[int], Form(description="Random seed")] = None,
    width: Annotated[int, Form(ge=256, le=2048)] = settings.default_width,
    height: Annotated[int, Form(ge=256, le=2048)] = settings.default_height,
    num_frames: Annotated[int, Form(ge=9, le=257)] = settings.default_num_frames,
    frame_rate: Annotated[float, Form(ge=1.0, le=60.0)] = settings.default_frame_rate,
    num_inference_steps: Annotated[int, Form(ge=1, le=100)] = settings.default_num_inference_steps,
    cfg_guidance_scale: Annotated[float, Form(ge=1.0, le=20.0)] = settings.default_cfg_guidance_scale,
    enhance_prompt: Annotated[bool, Form()] = True,
    image_strength: Annotated[float, Form(ge=0.0, le=1.0)] = 1.0,
    image_frame: Annotated[int, Form(ge=0)] = 0,
):
    """
    Generate a video from an input image.

    Uses TI2VidTwoStagesPipeline for production quality:
    - Stage 1: Generate at base resolution (e.g., 512x896)
    - Stage 2: 2x spatial upsampling (e.g., 1024x1792 output)

    Default settings produce 9:16 vertical video at 25fps.
    Note: dimensions must be divisible by 64 for the two-stage pipeline.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Please download models first.",
        )

    # Validate image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Try to acquire semaphore (non-blocking check)
    if not generation_semaphore._value:
        raise HTTPException(
            status_code=429,
            detail="Server is busy. Please try again later.",
        )

    await generation_semaphore.acquire()

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded image
    image_ext = Path(image.filename).suffix if image.filename else ".png"
    image_path = Path(settings.temp_dir) / f"{job_id}{image_ext}"

    try:
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
    except Exception as e:
        generation_semaphore.release()
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    # Create job record
    request = GenerationRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        cfg_guidance_scale=cfg_guidance_scale,
        enhance_prompt=enhance_prompt,
        image_strength=image_strength,
        image_frame=image_frame,
    )

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow().isoformat(),
    )

    # Start background generation
    background_tasks.add_task(run_generation, job_id, str(image_path), request)

    return GenerationResponse(
        job_id=job_id,
        status="pending",
        message="Video generation started",
        created_at=jobs[job_id].created_at,
    )


@app.post("/generate/base64", response_model=GenerationResponse)
async def generate_video_base64(
    background_tasks: BackgroundTasks,
    request: dict,
):
    """
    Generate a video from a base64-encoded image.

    Request body:
    {
        "image": "base64-encoded-image-data",
        "prompt": "your prompt",
        ...other GenerationRequest fields
    }
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Please download models first.",
        )

    if "image" not in request:
        raise HTTPException(status_code=400, detail="Missing 'image' field")

    if "prompt" not in request:
        raise HTTPException(status_code=400, detail="Missing 'prompt' field")

    # Try to acquire semaphore
    if not generation_semaphore._value:
        raise HTTPException(
            status_code=429,
            detail="Server is busy. Please try again later.",
        )

    await generation_semaphore.acquire()

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Decode and save image
    try:
        image_data = base64.b64decode(request["image"])
        image_path = Path(settings.temp_dir) / f"{job_id}.png"
        with open(image_path, "wb") as f:
            f.write(image_data)
    except Exception as e:
        generation_semaphore.release()
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Create generation request
    gen_request = GenerationRequest(
        prompt=request["prompt"],
        negative_prompt=request.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted",
        ),
        seed=request.get("seed"),
        width=request.get("width", settings.default_width),
        height=request.get("height", settings.default_height),
        num_frames=request.get("num_frames", settings.default_num_frames),
        frame_rate=request.get("frame_rate", settings.default_frame_rate),
        num_inference_steps=request.get("num_inference_steps", settings.default_num_inference_steps),
        cfg_guidance_scale=request.get("cfg_guidance_scale", settings.default_cfg_guidance_scale),
        enhance_prompt=request.get("enhance_prompt", True),
        image_strength=request.get("image_strength", 1.0),
        image_frame=request.get("image_frame", 0),
    )

    # Create job record
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow().isoformat(),
    )

    # Start background generation
    background_tasks.add_task(run_generation, job_id, str(image_path), gen_request)

    return GenerationResponse(
        job_id=job_id,
        status="pending",
        message="Video generation started",
        created_at=jobs[job_id].created_at,
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/videos/{job_id}")
async def get_video(job_id: str):
    """Download a generated video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready. Job status: {job.status}",
        )

    video_path = Path(settings.output_dir) / f"{job_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete video file if exists
    video_path = Path(settings.output_dir) / f"{job_id}.mp4"
    if video_path.exists():
        video_path.unlink()

    # Remove job record
    del jobs[job_id]

    return {"message": "Job deleted successfully"}


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 100,
):
    """List all jobs, optionally filtered by status."""
    result = list(jobs.values())

    if status:
        result = [j for j in result if j.status == status]

    # Sort by created_at descending
    result.sort(key=lambda x: x.created_at, reverse=True)

    return {"jobs": result[:limit], "total": len(result)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
