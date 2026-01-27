# LTX-2 Video Generation API

> LLM Instructions: This is an API for generating videos with synchronized audio using the LTX-2 19B model. Use POST /generate with a JSON body. Supports text-to-video, image-to-video, and audio-to-video conditioning.

## Quick Start

```bash
# Simple text-to-video with audio (9:16 portrait mobile)
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat sitting calmly on a cushion", "width": 288, "height": 512}'

# Best quality (mobile, ~2 seconds)
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano, studio lighting, camera pans slowly",
    "width": 288,
    "height": 512,
    "num_frames": 49,
    "num_inference_steps": 40,
    "guidance_scale": 3.0
  }'

# Quick preview (test prompts fast)
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano",
    "width": 288,
    "height": 512,
    "num_frames": 9,
    "num_inference_steps": 5
  }'

# Image-to-video (start frame)
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The scene animates with gentle movement",
    "width": 288,
    "height": 512,
    "image_start": "data:image/png;base64,<BASE64_IMAGE>"
  }'

# Image-to-video (start + end frame interpolation)
curl -X POST http://<SERVER>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Smooth transition between two scenes",
    "width": 288,
    "height": 512,
    "image_start": "data:image/png;base64,<START_IMAGE>",
    "image_end": "data:image/png;base64,<END_IMAGE>"
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
| 9 | 288x512 | 5 | 0.36s | ~13s | ~40 KB | text (preview) |
| 25 | 288x512 | 20 | 1.0s | ~18s | ~100 KB | text |
| 49 | 288x512 | 30 | 1.96s | ~21s | ~280 KB | text (balanced) ⭐ |
| 49 | 288x512 | 40 | 1.96s | ~28s | ~280 KB | text (best quality) ⭐ |
| 121 | 288x512 | 40 | 4.84s | ~28s | ~800 KB | text (long) |

**Recommended Defaults:**
- **Mobile/TikTok:** width=288, height=512, frames=49, steps=40, guidance=3.0
- **Quick Preview:** frames=9, steps=5
- **Long Video:** frames=121, steps=40

**Memory Usage:**
- Peak GPU memory: ~31GB during inference
- Model checkpoint: ~19GB (FP8 quantized)
- Text encoder (Gemma 3): ~27GB (loaded/unloaded per request)

## Best Practices for Quality

### 1. Resolution for Your Use Case
| Use Case | Width | Height | Notes |
|----------|-------|--------|-------|
| TikTok/Shorts | 288 | 512 | **Recommended** - 9:16 portrait |
| HD Mobile | 384 | 672 | Higher quality mobile |
| YouTube/Landscape | 768 | 432 | 16:9 landscape |
| Instagram | 512 | 512 | 1:1 square |

### 2. Frame Count & Duration
| Frames | Duration | Use Case |
|--------|----------|----------|
| 9 | 0.36s | **Quick preview/test prompts** |
| 25 | 1.0s | Short clips, fast testing |
| 49 | 1.96s | **Balanced quality/speed** ⭐ |
| 121 | 4.84s | Long clips, maximum quality |

### 3. Inference Steps (Quality vs Speed)
| Steps | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| 5 | Low | Fast (~13s) | Quick previews only |
| 20 | Medium | Medium (~18s) | Good for testing |
| 30 | Good | Medium (~21s) | **Recommended balance** ⭐ |
| 40 | Excellent | Slow (~28s) | **Best quality** ⭐ |
| 50 | Max | Very slow | Final output, high detail |

### 4. Guidance Scale
| Value | Effect |
|-------|--------|
| 1.0 | Disables CFG (follows prompt loosely) |
| 2.0 | Low guidance |
| **3.0** | **Recommended sweet spot** |
| 4.0+ | High guidance (can look over-processed) |

### 5. Prompt Engineering
**Be descriptive about BOTH visuals AND audio:**
```
Good: "A cat sitting calmly on a cushion, soft lighting, gentle purring sounds"
Bad: "A cat"
```

**Mention camera movement:**
```
"camera pans slowly across the scene"
"static shot, focused on the subject"
"handheld camera, slight movement"
```

**Include lighting details:**
```
"soft natural lighting from window"
"dramatic shadows, cinematic lighting"
"golden hour sunset lighting"
```

**For speech generation:**
```
"A person speaking: [dialogue text], [scene description]"
```

**For ambient audio:**
```
"A waterfall in a forest with rushing water sounds and birds chirping"
```

### 6. Image Conditioning (Character Consistency)
Use `image_start` for consistent character appearance:

```json
{
  "image_start": "base64_reference_image",
  "image_strength": 1.0,  // 0.0-1.0, higher = more consistency
  "prompt": "The scene comes alive with gentle movement"
}
```

**Tips:**
- Use high-quality reference images
- Set `image_strength: 1.0` for maximum consistency
- Keep seed consistent for similar expressions
- Use 40+ inference steps for best quality

### 7. Seed Control
Use seeds for reproducibility:
```json
{
  "seed": 1234567890
}
```

**Strategy:**
1. Test with random seeds until you find a good one
2. Save that seed for consistent results
3. Vary seeds to explore different interpretations

### 8. Workflow Recommendations

**For Character Videos (e.g., Aiko):**
1. Generate high-quality base image
2. Use as `image_start` with `image_strength: 1.0`
3. Keep seed consistent
4. Use 40+ inference steps

**For Product Demos:**
1. Start with product image as `image_start`
2. Describe motion in prompt
3. Use 30-40 inference steps
4. Test with 9 frames first, then scale up

**For Backgrounds/Scenery:**
1. Be very descriptive
2. Mention camera movement
3. Use 49-121 frames for smooth motion
4. Try multiple seeds for different compositions

---

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
  "image_start": null,
  "image_end": null,
  "image_strength": 1.0,
  "include_audio": true,
  "audio": null,
  "audio_strength": 1.0
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Video description. Include "A person speaking: [dialogue]" for speech. **Be descriptive about visuals AND audio.** |
| `negative_prompt` | string | "blurry..." | What to avoid |
| `width` | int | 288 | Must be divisible by 32. **288 recommended for 9:16 mobile/TikTok** |
| `height` | int | 512 | Must be divisible by 32. **512 recommended for 9:16 mobile/TikTok** |
| `num_frames` | int | 49 | Number of frames (49=~2s, 121=~5s at 24fps). Use 9 for quick previews. |
| `fps` | int | 25 | Frame rate |
| `guidance_scale` | float | 3.0 | CFG scale (1.0 disables CFG, **3.0 recommended** as sweet spot) |
| `num_inference_steps` | int | 40 | Quality steps (**5=preview**, **30=good**, **40=excellent**, 50=max). |
| `seed` | int | random | For reproducibility. Save good seeds! |
| `image_start` | string | null | Base64 image for first frame conditioning (use for character consistency) |
| `image_end` | string | null | Base64 image for last frame conditioning |
| `image_strength` | float | 1.0 | Image conditioning strength (0.0-1.0, **1.0=max adherence** to start image) |
| `include_audio` | bool | true | Include generated audio in output |
| `audio` | string | null | Base64 WAV audio for audio-to-video conditioning |
| `audio_strength` | float | 1.0 | Audio conditioning strength (0.0-1.0) |

### Response

```json
{
  "video": "data:video/mp4;base64,...",
  "duration_seconds": 4.84,
  "resolution": "288x512",
  "fps": 25,
  "seed": 1234567890,
  "generation_time_seconds": 27.72,
  "has_audio": true,
  "has_image_start": false,
  "has_image_end": false
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
                   steps=30, image_start_path=None, image_end_path=None,
                   image_strength=1.0, audio_path=None):
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "include_audio": True,
    }

    if image_start_path:
        with open(image_start_path, "rb") as f:
            payload["image_start"] = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        payload["image_strength"] = image_strength

    if image_end_path:
        with open(image_end_path, "rb") as f:
            payload["image_end"] = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

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
    print(f"  Generation time: {result['generation_time_seconds']}s")
    print(f"  Has audio: {result['has_audio']}")
    print(f"  Image start: {result.get('has_image_start', False)}")
    print(f"  Image end: {result.get('has_image_end', False)}")

    return result

# Text-to-video
generate_video("A cat sitting calmly on a cushion, soft lighting")

# Image-to-video (start frame only)
generate_video("The scene comes alive with gentle movement",
               image_start_path="start.png")

# Image-to-video (start + end frame interpolation)
generate_video("Smooth transition between two scenes",
               image_start_path="start.png",
               image_end_path="end.png")

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
