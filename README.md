# LTX-2 Image-to-Video API

Docker-based API for generating 9:16 1080p (1080x1920) vertical videos from images using [Lightricks LTX-2](https://github.com/Lightricks/LTX-2) model.

## Features

- **9:16 Vertical Video**: Optimized for 1080x1920 portrait videos (TikTok, Reels, Shorts)
- **Image-to-Video**: Generate videos conditioned on input images
- **Two-Stage Pipeline**: High-quality output with spatial upsampling
- **FP8 Quantization**: Lower VRAM requirements (~40GB recommended)
- **Async API**: Background job processing with status tracking
- **REST API**: Easy integration with any application

## Requirements

- NVIDIA GPU with 40GB+ VRAM (A100, H100, or similar)
- Docker with NVIDIA Container Toolkit
- ~50GB disk space for models

## Quick Start

### 1. Clone the repository

```bash
git clone <this-repo>
cd ltx2-api
```

### 2. Download models

Set your HuggingFace token (required for Gemma access):

```bash
export HF_TOKEN=your_huggingface_token
```

Download models using Docker:

```bash
docker-compose --profile download run model-downloader
```

Or manually:

```bash
pip install huggingface_hub
python scripts/download_models.py --models-dir ./models
```

### 3. Start the API server

```bash
docker-compose up -d
```

### 4. Generate videos

```bash
# Health check
curl http://localhost:8000/health

# Generate video from image
curl -X POST http://localhost:8000/generate \
  -F "image=@your_image.jpg" \
  -F "prompt=A person smiling and waving at the camera"

# Check job status
curl http://localhost:8000/jobs/{job_id}

# Download video
curl http://localhost:8000/videos/{job_id} -o output.mp4
```

## API Endpoints

### `POST /generate`

Generate a video from an uploaded image.

**Form Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | File | Required | Input image (PNG, JPG) |
| `prompt` | string | Required | Text prompt describing the video |
| `negative_prompt` | string | "worst quality..." | Negative prompt |
| `seed` | int | Random | Seed for reproducibility |
| `width` | int | 1080 | Video width (256-2048) |
| `height` | int | 1920 | Video height (256-2048) |
| `num_frames` | int | 121 | Number of frames (9-257) |
| `frame_rate` | float | 25.0 | Frames per second |
| `num_inference_steps` | int | 30 | Denoising steps |
| `cfg_guidance_scale` | float | 3.0 | CFG scale |
| `enhance_prompt` | bool | true | Use prompt enhancement |
| `image_strength` | float | 1.0 | Image conditioning strength |
| `image_frame` | int | 0 | Frame to place image |

### `POST /generate/base64`

Generate video from base64-encoded image (JSON body).

```json
{
  "image": "base64-encoded-image",
  "prompt": "your prompt",
  "width": 1080,
  "height": 1920
}
```

### `GET /jobs/{job_id}`

Get job status.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "video_url": "/videos/uuid",
  "created_at": "2024-01-01T00:00:00",
  "completed_at": "2024-01-01T00:01:00"
}
```

### `GET /videos/{job_id}`

Download the generated video file.

### `GET /jobs`

List all jobs.

### `DELETE /jobs/{job_id}`

Delete a job and its video.

### `GET /health`

Health check endpoint.

## Configuration

Environment variables (set in docker-compose.yml):

| Variable | Default | Description |
|----------|---------|-------------|
| `LTX_CHECKPOINT_PATH` | /models/ltx-2-19b-distilled-fp8.safetensors | Model checkpoint |
| `LTX_GEMMA_ROOT` | /models/gemma-3-4b-it | Gemma text encoder path |
| `LTX_SPATIAL_UPSAMPLER_PATH` | /models/ltx-2-spatial-upsampler-x2-1.0.safetensors | Upsampler model |
| `LTX_DEFAULT_WIDTH` | 1080 | Default video width |
| `LTX_DEFAULT_HEIGHT` | 1920 | Default video height |
| `LTX_DEFAULT_NUM_FRAMES` | 121 | Default frame count (~5s at 25fps) |
| `LTX_ENABLE_FP8` | true | Use FP8 quantization |
| `LTX_MAX_CONCURRENT_JOBS` | 1 | Max parallel generations |

## Model Files

After downloading, your `./models` directory should contain:

```
models/
├── ltx-2-19b-distilled-fp8.safetensors  # Main model (~19GB)
├── ltx-2-spatial-upsampler-x2-1.0.safetensors  # Upsampler
├── ltx-2-19b-distilled-lora.safetensors  # Distilled LoRA
└── gemma-3-4b-it/  # Text encoder (~8GB)
    ├── config.json
    ├── model.safetensors
    └── ...
```

## GPU Memory Requirements

| Model | VRAM Required |
|-------|---------------|
| Distilled FP8 (recommended) | ~40GB |
| Distilled | ~60GB |
| Dev FP8 | ~50GB |
| Dev (full) | ~80GB |

## Prompt Tips

Write detailed, chronological descriptions under 200 words:

```
Good: "A young woman with long dark hair stands in a sunlit garden.
She slowly turns to face the camera with a warm smile.
The camera gently zooms in as flower petals drift past.
Soft golden hour lighting creates a dreamy atmosphere."

Bad: "woman in garden smiling"
```

## Example Python Client

```python
import requests
import time

# Generate video
with open("input.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/generate",
        files={"image": f},
        data={
            "prompt": "A person slowly turning to face the camera with a warm smile",
            "width": 1080,
            "height": 1920,
        }
    )

job = response.json()
job_id = job["job_id"]

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"Failed: {status['error']}")
        break
    time.sleep(5)

# Download video
video = requests.get(f"http://localhost:8000/videos/{job_id}")
with open("output.mp4", "wb") as f:
    f.write(video.content)
```

## Troubleshooting

### Out of Memory
- Use the FP8 checkpoint
- Reduce `num_frames`
- Reduce resolution
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Slow Generation
- Reduce `num_inference_steps` to 20-25
- Use DistilledPipeline for faster inference

### Model Download Issues
- Accept Gemma license at https://huggingface.co/google/gemma-3-4b-it
- Set `HF_TOKEN` environment variable

## License

LTX-2 model is released under the LTX-2 Community License. See [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) for details.
