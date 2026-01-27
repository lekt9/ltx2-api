# LTX-2 Video Generation API - Best Practices Addendum

> This file supplements README.md with detailed best practices for getting optimal quality from LTX-2.

---

## Quick Reference Cheat Sheet

### Best Quality Settings (Mobile/TikTok) ⭐
```json
{
  "num_frames": 49,
  "num_inference_steps": 40,
  "guidance_scale": 3.0,
  "width": 288,
  "height": 512
}
```

### Long Video Settings (5 seconds)
```json
{
  "num_frames": 121,
  "num_inference_steps": 40,
  "guidance_scale": 3.0,
  "width": 288,
  "height": 512
}
```

### Fast Preview Settings (Test prompts)
```json
{
  "num_frames": 9,
  "num_inference_steps": 5,
  "width": 288,
  "height": 512
}
```

### Balanced Settings (Good quality, reasonable speed)
```json
{
  "num_frames": 49,
  "num_inference_steps": 30,
  "guidance_scale": 3.0,
  "width": 288,
  "height": 512
}
```

### Character Consistency (e.g., Aiko)
```json
{
  "image_start": "base64_reference_image",
  "image_strength": 1.0,
  "num_frames": 49,
  "num_inference_steps": 40,
  "guidance_scale": 3.0,
  "width": 288,
  "height": 512,
  "seed": 1234567890
}
```

---

## Best Practices for Quality

### 1. Resolution for Your Use Case
| Use Case | Width | Height | Aspect Ratio | Notes |
|----------|-------|--------|--------------|-------|
| TikTok/Shorts | 288 | 512 | 9:16 portrait | **Recommended default** |
| HD Mobile | 384 | 672 | 9:16 portrait | Higher quality mobile |
| YouTube/Landscape | 768 | 432 | 16:9 landscape | Standard video |
| Instagram | 512 | 512 | 1:1 square | Square format |

### 2. Frame Count & Duration
| Frames | Duration at 24fps | Use Case |
|--------|------------------|----------|
| 9 | 0.36s | Quick preview/test prompts |
| 25 | 1.0s | Short clips, fast testing |
| 49 | 1.96s | **Balanced quality/speed** ⭐ |
| 121 | 4.84s | Long clips, maximum quality |

### 3. Inference Steps (Quality vs Speed)
| Steps | Quality | Generation Time | Use Case |
|-------|---------|-----------------|----------|
| 5 | Low | ~13s | Quick previews only |
| 20 | Medium | ~18s | Good for testing |
| 30 | Good | ~21s | **Recommended balance** ⭐ |
| 40 | Excellent | ~28s | **Best quality** ⭐ |
| 50 | Max | Very slow | Final output, high detail |

### 4. Guidance Scale (CFG)
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
"camera zooms in gradually"
```

**Include lighting details:**
```
"soft natural lighting from window"
"dramatic shadows, cinematic lighting"
"golden hour sunset lighting"
"studio lighting, bright and clean"
```

**For speech generation:**
```
"A person speaking: [dialogue text], [scene description]"
```

Example:
```
"A person speaking: Hello and welcome to the presentation. The speaker stands in front of a whiteboard in a modern office."
```

**For ambient audio:**
```
"A waterfall in a forest with rushing water sounds and birds chirping"
"Ocean waves crashing on a rocky shore at sunset"
"Coffee shop ambience with quiet conversation"
```

### 6. Image Conditioning (Character Consistency)

Use `image_start` for consistent character appearance across videos:

```json
{
  "image_start": "base64_reference_image",
  "image_strength": 1.0,
  "prompt": "The scene comes alive with gentle movement"
}
```

**Tips for character consistency:**
- Use high-quality reference images (clear face, good lighting)
- Set `image_strength: 1.0` for maximum consistency
- Keep seed consistent across generations for same look
- Use 40+ inference steps for best quality
- Describe camera movement in prompt for natural animation

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
4. Same prompt + different seed = very different results

### 8. Workflow Recommendations

**For Character Videos (e.g., Aiko):**
1. Generate high-quality base image of character
2. Use as `image_start` with `image_strength: 1.0`
3. Keep seed consistent for same expressions
4. Use 40+ inference steps for best quality
5. Describe camera movement and lighting

**For Product Demos:**
1. Start with product image as `image_start`
2. Describe the motion you want in prompt
3. Use 30-40 inference steps
4. Test with 9 frames first, then scale up
5. Mention camera angle and lighting

**For Backgrounds/Scenery:**
1. Be very descriptive in prompt
2. Mention camera movement (pan, zoom)
3. Use 49-121 frames for smooth motion
4. Include audio description (soundscape)
5. Try multiple seeds for different compositions

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `num_frames` (try 25 or 49 instead of 121)
- Reduce resolution (use 288x512 - recommended default)
- Reduce `num_inference_steps` (try 20 instead of 40)

### Slow Generation
- First request loads models (~20-30s overhead) - normal behavior
- Subsequent requests are faster
- Use fewer inference steps (9 frames, 5 steps) for previews

### No Audio in Output
- Set `include_audio: true` in request
- Audio is generated automatically by the model

### Video Quality Issues
- Increase `num_inference_steps` to 40-50 for best quality
- Use more descriptive prompts (visuals + audio)
- Try different seeds (same prompt = very different results)
- Use `image_start` with good reference image for consistency

### Uncanny/Face Issues
- Use `image_start` with high-quality face reference
- Adjust `image_strength` (0.5-1.0)
- Try different seeds
- Simplify the prompt

### Character Inconsistency
- Use `image_start` with reference image
- Set `image_strength: 1.0` for maximum adherence
- Keep seed consistent across generations
- Use 40+ inference steps
- Ensure reference image is high quality

---

## Key Takeaways

1. **More steps = better quality** (use 30-40 for good results, 40+ for best)
2. **Describe visuals AND audio** in prompts
3. **Use image conditioning** for character consistency
4. **Test with 9 frames first** before full generation
5. **Try different seeds** - same prompt = very different results
6. **Guidance scale 3.0** is the sweet spot
7. **288x512 is recommended** for mobile/TikTok content
8. **Save good seeds** for reproducibility
