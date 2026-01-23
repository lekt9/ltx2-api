# LTX-2 Video Generation API

> LLM Instructions: This is an API for generating videos with synchronized audio using the LTX-2 19B model. Use POST /generate with a JSON body. Supports text-to-video, image-to-video, and audio-to-video conditioning.

## Quick Start

```bash
# Simple text-to-video with audio (9:16 portrait)
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat sitting calmly on a cushion", "width": 288, "height": 512}'

# Longer video with more frames
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano, studio lighting",
    "width": 288,
    "height": 512,
    "num_frames": 121,
    "num_inference_steps": 40
  }'

# Image-to-video
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The scene animates with gentle movement",
    "width": 288,
    "height": 512,
    "image": "data:image/png;base64,<BASE64_IMAGE>"
  }'
```

## Audio Capabilities

| Feature | Status |
|---------|--------|
| Audio OUTPUT | YES - generates synchronized audio (24kHz stereo) |
| Audio INPUT | YES - can condition video generation on audio input |

The model jointly generates video AND audio from text/image/audio prompts.

### Audio-to-Video Example
```bash
# Convert audio to base64
AUDIO_B64=$(base64 -w0 input.wav)

# Generate video conditioned on audio
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"A person speaking in front of a camera\",
    \"width\": 288,
    \"height\": 512,
    \"audio\": \"data:audio/wav;base64,$AUDIO_B64\"
  }"
```

**Audio Input Requirements:**
- Format: WAV file (base64-encoded)
- Sample rate: Any (resampled internally to 16kHz)
- Channels: Mono or stereo
- Length: Any duration (automatically resampled to match video length)

## Performance Benchmarks

Tested on NVIDIA B200 (178GB VRAM):

| Frames | Resolution | Steps | Duration | Gen Time | File Size | Mode |
|--------|------------|-------|----------|----------|-----------|------|
| 9 | 288x512 | 5 | 0.36s | ~13s | ~40 KB | text |
| 25 | 288x512 | 20 | 1.0s | ~18s | ~100 KB | text |
| 25 | 288x512 | 20 | 1.0s | ~18s | ~280 KB | audio |
| 49 | 288x512 | 30 | 1.96s | ~21s | ~280 KB | text |
| 121 | 288x512 | 40 | 4.84s | ~28s | ~800 KB | text |

**Memory Usage:**
- Peak GPU memory: ~31GB during inference
- Model checkpoint: ~19GB (FP8 quantized)
- Text encoder (Gemma 3): ~27GB (loaded/unloaded per request)

## API Reference

### Endpoint
```
POST /generate
Content-Type: application/json
```

### Request Body

```json
{
  "prompt": "required - describe the video and any speech/sounds",
  "negative_prompt": "blurry, low quality, distorted, glitchy, watermark",
  "width": 288,
  "height": 512,
  "num_frames": 121,
  "fps": 25,
  "guidance_scale": 3.0,
  "num_inference_steps": 40,
  "seed": null,
  "image": null,
  "image_cond_noise_scale": 0.15,
  "include_audio": true,
  "audio": null,
  "audio_cond_noise_scale": 0.15
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Video description. Include "A person speaking: [dialogue]" for speech |
| `negative_prompt` | string | "blurry..." | What to avoid |
| `width` | int | 768 | Must be divisible by 32 (recommend 288 for 9:16) |
| `height` | int | 512 | Must be divisible by 32 |
| `num_frames` | int | 121 | Number of frames (~5 sec at 25fps) |
| `fps` | int | 25 | Frame rate |
| `guidance_scale` | float | 3.0 | CFG scale (1.0 disables CFG, 3.0 recommended) |
| `num_inference_steps` | int | 40 | Quality steps (20-50, higher = better quality) |
| `seed` | int | random | For reproducibility |
| `image` | string | null | Base64 image for image-to-video mode |
| `image_cond_noise_scale` | float | 0.15 | How much to deviate from source image (0=exact, 1=ignore) |
| `include_audio` | bool | true | Include generated audio in output |
| `audio` | string | null | Base64 WAV audio for audio-to-video conditioning |
| `audio_cond_noise_scale` | float | 0.15 | How much to deviate from source audio (0=exact, 1=ignore) |

### Response

```json
{
  "video": "data:video/mp4;base64,...",
  "duration_seconds": 4.84,
  "resolution": "288x512",
  "fps": 25,
  "seed": 1234567890,
  "generation_time_seconds": 27.72,
  "mode": "text-to-video | image-to-video | audio-to-video",
  "has_audio": true
}
```

## Aspect Ratios (divisible by 32)

| Aspect | Width | Height | Use Case | Tested |
|--------|-------|--------|----------|--------|
| 9:16 portrait | 288 | 512 | Mobile/TikTok | Yes |
| 9:16 portrait HD | 384 | 672 | Higher quality mobile | - |
| 16:9 landscape | 768 | 432 | Standard video | - |
| 1:1 square | 512 | 512 | Instagram | - |

**Note:** 9:16 portrait (288x512) is the recommended and tested configuration.

## Prompt Tips

### For Speech
```
"A person speaking: Hello, welcome to the presentation. The speaker stands in front of a whiteboard."
```

### For Ambient Audio
```
"A waterfall in a forest with rushing water sounds and birds chirping"
"Ocean waves crashing on a rocky shore at sunset"
```

### For Music (lower quality)
```
"A piano playing a gentle melody in a concert hall"
```

### General Tips
- Be descriptive about both visuals AND sounds
- Mention camera movement: "camera pans slowly", "static shot"
- Include lighting: "soft lighting", "dramatic shadows", "golden hour"

## Python Client

```python
import requests
import base64

def generate_video(prompt, server="http://localhost:8000",
                   width=288, height=512, num_frames=49,
                   steps=30, image_path=None, audio_path=None):
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "include_audio": True,
    }

    if image_path:
        with open(image_path, "rb") as f:
            payload["image"] = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    if audio_path:
        with open(audio_path, "rb") as f:
            payload["audio"] = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode()}"

    response = requests.post(f"{server}/generate", json=payload, timeout=300)
    result = response.json()

    if "error" in result or "detail" in result:
        raise Exception(result.get("error") or result.get("detail"))

    # Save video
    video_data = result["video"].split(",")[1]
    output_file = "output.mp4"
    with open(output_file, "wb") as f:
        f.write(base64.b64decode(video_data))

    print(f"Saved: {output_file}")
    print(f"  Duration: {result['duration_seconds']}s")
    print(f"  Generation time: {result['generation_time_seconds']}s")
    print(f"  Has audio: {result['has_audio']}")

    return result

# Text-to-video
generate_video("A cat sitting calmly on a cushion, soft lighting")

# Image-to-video
generate_video("The scene comes alive with gentle movement",
               image_path="input.png")

# Audio-to-video (sync video to audio input)
generate_video("A person speaking in front of a camera",
               audio_path="speech.wav")

# Quick preview (fewer frames/steps)
generate_video("A quick test", num_frames=9, steps=5)
```

## Bash Script

```bash
#!/bin/bash
SERVER="${LTX_SERVER:-http://localhost:8000}"

generate_video() {
    local prompt="$1"
    local output="${2:-output.mp4}"

    curl -s --max-time 300 -X POST "$SERVER/generate" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"width\": 288, \"height\": 512}" \
    | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
if 'error' in data or 'detail' in data:
    print(f'Error: {data}')
    sys.exit(1)
video = data['video'].split(',')[1]
with open('$output', 'wb') as f:
    f.write(base64.b64decode(video))
print(f'Saved: $output ({data[\"duration_seconds\"]}s, gen_time={data[\"generation_time_seconds\"]}s)')
"
}

# Usage
generate_video "A cat playing piano" "cat_piano.mp4"
```

## Other Endpoints

```bash
# Health check
curl http://<SERVER>:8000/health
# Returns: {"status": "healthy"}

# API info
curl http://<SERVER>:8000/
# Returns: API description and defaults

# Interactive docs (Swagger UI)
# Open in browser: http://<SERVER>:8000/docs
```

## Docker

### Build
```bash
docker build -t ltx2-worker .
```

### Run as HTTP Server
```bash
docker run -d --gpus all \
  --name ltx2-server \
  -p 8000:8000 \
  -v /runpod-volume:/runpod-volume \
  ltx2-worker python /app/server.py
```

### Run as RunPod Serverless Worker
```bash
docker run -d --gpus all \
  -v /runpod-volume:/runpod-volume \
  ltx2-worker python /app/handler.py
```

### Check Logs
```bash
docker logs -f ltx2-server
```

### Stop/Remove
```bash
docker stop ltx2-server && docker rm ltx2-server
```

## Hardware Requirements

| GPU | VRAM | Status |
|-----|------|--------|
| NVIDIA B200 | 178GB | Tested, works |
| NVIDIA H100 | 80GB | Should work |
| NVIDIA A100 | 80GB | Should work |
| NVIDIA A100 | 40GB | May need lower resolution |

**Memory breakdown:**
- Transformer (FP8): ~19GB
- Text encoder (Gemma 3): ~27GB (loaded/unloaded)
- Video encoder/decoder: ~2GB
- Audio encoder (for audio input): ~1GB (loaded/unloaded)
- Audio decoder + vocoder: ~2GB
- Peak inference: ~31GB

## Model Info

- **Model**: Lightricks/LTX-2 (19B parameters)
- **Checkpoint**: `ltx-2-19b-distilled-fp8.safetensors` (distilled, FP8 quantized)
- **Architecture**: DiT (Diffusion Transformer) with dual video/audio streams
- **Audio**: 24kHz stereo, HiFi-GAN vocoder
- **Text Encoder**: Gemma 3 (13B parameters)
- **Video Format**: H.264/AAC in MP4 container

## Troubleshooting

### CUDA Out of Memory
- Reduce `num_frames` (try 25 or 49 instead of 121)
- Reduce resolution (use 288x512)
- Reduce `num_inference_steps` (try 20 instead of 40)

### Slow Generation
- First request loads models (~20-30s overhead)
- Subsequent requests are faster
- Use fewer inference steps for previews

### No Audio in Output
- Set `include_audio: true` in request
- Audio is generated automatically, cannot be disabled per-frame

### Video Quality Issues
- Increase `num_inference_steps` (40-50 for best quality)
- Use more descriptive prompts
- Try different seeds

## Files

```
/app/
├── handler.py      # RunPod serverless handler + generation logic
├── server.py       # FastAPI HTTP server wrapper
├── Dockerfile      # Container build configuration
├── llms.txt        # This documentation
└── test_input.json # Example input for testing
```
