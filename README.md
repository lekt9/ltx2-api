# LTX-2 Video Generation - RunPod Serverless Worker

Generate high-quality AI videos using LTX-2 on RunPod Serverless with H100 GPU.

## Features

- **Fast**: ~60-80 seconds for a 4-second 720p video on H100
- **Pay-per-use**: Only charged when generating (scales to zero)
- **Simple API**: Just send a prompt, get a video
- **Cost-effective**: ~$0.05-0.10 per 4-second video

## Deployment

### Option 1: Deploy via GitHub (Recommended)

1. Push this repo to GitHub
2. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
3. Click "New Endpoint" → "Deploy from GitHub"
4. Connect your repo
5. Configure:
   - **GPU**: H100 SXM (80GB) - fastest
   - **Min Workers**: 0 (scales to zero)
   - **Max Workers**: 3
   - **Idle Timeout**: 5 seconds
   - **Volume**: 100GB (for model cache)

### Option 2: Deploy via Docker Hub

```bash
# Build locally
docker build -t your-dockerhub/ltx2-worker:latest .

# Push to Docker Hub
docker push your-dockerhub/ltx2-worker:latest
```

Then in RunPod:
1. Create new endpoint
2. Use your Docker image URL
3. Configure GPU as H100 SXM

### Option 3: Deploy via RunPod CLI

```bash
# Install runpodctl
brew install runpod/tap/runpodctl

# Login
runpodctl login

# Deploy
runpodctl deploy --name ltx2-worker --image your-image:latest --gpu H100
```

## Environment Variables

Set these in RunPod dashboard:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Optional | HuggingFace token (if using gated models) |

## API Usage

### Endpoint URL
```
https://api.runpod.ai/v2/{YOUR_ENDPOINT_ID}/runsync
```

### Request Format

```bash
curl -X POST \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cat playing piano, studio lighting, 4K",
      "width": 1280,
      "height": 720,
      "num_frames": 97
    }
  }' \
  "https://api.runpod.ai/v2/$RUNPOD_LTX_ENDPOINT_ID/runsync"
```

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text description of the video |
| `negative_prompt` | string | "blurry, low quality..." | What to avoid |
| `width` | int | 1280 | Video width (720, 1280, 1920) |
| `height` | int | 720 | Video height (480, 720, 1080) |
| `num_frames` | int | 97 | Number of frames (~4 sec at 24fps) |
| `fps` | int | 24 | Frames per second |
| `guidance_scale` | float | 7.5 | Prompt adherence (5-15) |
| `num_inference_steps` | int | 30 | Quality steps (20-50) |
| `seed` | int | random | Reproducibility seed |

### Response Format

```json
{
  "id": "job-uuid",
  "status": "COMPLETED",
  "output": {
    "video": "data:video/mp4;base64,AAAA...",
    "duration_seconds": 4.04,
    "resolution": "1280x720",
    "fps": 24,
    "seed": 12345,
    "generation_time_seconds": 72.5
  }
}
```

### Async Usage

For longer videos, use async mode:

```bash
# Submit job
curl -X POST ... "https://api.runpod.ai/v2/$ENDPOINT_ID/run"
# Returns: {"id": "job-123", "status": "IN_QUEUE"}

# Check status
curl -H "Authorization: Bearer $API_KEY" \
  "https://api.runpod.ai/v2/$ENDPOINT_ID/status/job-123"
```

## Performance Guide

### Recommended Settings by Use Case

| Use Case | Resolution | Frames | Time | Cost |
|----------|------------|--------|------|------|
| Quick preview | 720p | 49 (~2s) | ~30s | ~$0.02 |
| Social media | 720p | 97 (~4s) | ~60s | ~$0.05 |
| High quality | 1080p | 97 (~4s) | ~90s | ~$0.08 |
| Long form | 720p | 241 (~10s) | ~4min | ~$0.18 |

### GPU Comparison

| GPU | 4s 720p | 4s 1080p | Cost/hr |
|-----|---------|----------|---------|
| RTX 4090 | ~45s | ~90s | $0.44 |
| A40 | ~60s | ~120s | $0.79 |
| H100 SXM | ~30s | ~60s | $2.71 |

H100 recommended for production - faster turnaround despite higher hourly cost.

## Claude Code Integration

This worker integrates with Claude Code via `/generate-video`:

```
/generate-video A majestic eagle soaring over mountains at sunset
```

Set environment variables:
```bash
export RUNPOD_API_KEY="your_api_key"
export RUNPOD_LTX_ENDPOINT_ID="your_endpoint_id"
```

## Troubleshooting

### Cold Start Slow?
First request after idle may take 2-5 minutes to load models. Subsequent requests are fast.

**Solution**: Set `Min Workers: 1` to keep one worker warm (costs ~$2/hr idle on H100).

### Out of Memory?
Reduce resolution or frames:
```json
{"input": {"prompt": "...", "width": 720, "height": 480, "num_frames": 49}}
```

### Video Quality Issues?
Increase steps and guidance:
```json
{"input": {"prompt": "...", "num_inference_steps": 50, "guidance_scale": 10}}
```

## License

LTX-2 is Apache 2.0 licensed. This worker code is provided as-is.
