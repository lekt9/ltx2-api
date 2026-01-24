"""
RunPod Serverless Handler for LTX-2 Video Generation

Supports text-to-video, image-to-video, and audio-to-video generation.
Generates synchronized audio alongside video.
"""

import base64
import io
import os
import subprocess
import tempfile
import time
from typing import Optional, List, Tuple

import numpy as np
import runpod
import torch
from scipy.io import wavfile

# Configuration
CACHE_DIR = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
MODEL_ID = "Lightricks/LTX-2"
AUDIO_SAMPLE_RATE = 24000

# Global pipeline cache
PIPELINE = None
AUDIO_ENCODER = None
AUDIO_PROCESSOR = None


def load_pipeline():
    """Load LTX-2 pipeline using TI2VidOneStagePipeline."""
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    print("Loading LTX-2 pipeline...")
    start = time.time()

    from huggingface_hub import hf_hub_download, snapshot_download
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

    # Download model files
    print("Downloading LTX-2 model files...")
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="ltx-2-19b-distilled-fp8.safetensors",
        cache_dir=CACHE_DIR,
    )
    gemma_root = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["text_encoder/*", "tokenizer/*"],
        cache_dir=CACHE_DIR,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Gemma root: {gemma_root}")

    # Initialize pipeline
    PIPELINE = TI2VidOneStagePipeline(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        loras=[],
        fp8transformer=True,
    )

    print(f"Pipeline initialized in {time.time() - start:.1f}s")
    return PIPELINE


def load_audio_encoder():
    """Load audio encoder for audio-to-video conditioning."""
    global AUDIO_ENCODER, AUDIO_PROCESSOR

    if AUDIO_ENCODER is not None:
        return AUDIO_ENCODER, AUDIO_PROCESSOR

    print("Loading audio encoder...")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from ltx_core.model.audio_vae.audio_vae import AudioEncoder
    from ltx_core.model.audio_vae.audio_processor import AudioProcessor

    checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="ltx-2-19b-distilled-fp8.safetensors",
        cache_dir=CACHE_DIR,
    )

    # Load audio encoder weights
    state_dict = load_file(checkpoint_path)
    audio_keys = {k: v for k, v in state_dict.items() if k.startswith("audio_vae.encoder.")}
    audio_state = {k.replace("audio_vae.encoder.", ""): v for k, v in audio_keys.items()}

    AUDIO_ENCODER = AudioEncoder()
    AUDIO_ENCODER.load_state_dict(audio_state)
    AUDIO_ENCODER.to("cuda", torch.bfloat16)
    AUDIO_ENCODER.eval()

    AUDIO_PROCESSOR = AudioProcessor()
    print("Audio encoder loaded")
    return AUDIO_ENCODER, AUDIO_PROCESSOR


def decode_base64_audio(audio_data: str) -> Tuple[torch.Tensor, int]:
    """Decode base64 WAV audio to tensor."""
    if audio_data.startswith("data:"):
        audio_data = audio_data.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_data)
    with io.BytesIO(audio_bytes) as f:
        sample_rate, waveform = wavfile.read(f)

    # Convert to float tensor
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    elif waveform.dtype == np.uint8:
        waveform = (waveform.astype(np.float32) - 128) / 128.0

    waveform = torch.from_numpy(waveform.copy())
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).repeat(2, 1)
    else:
        waveform = waveform.T

    waveform = waveform.unsqueeze(0)
    return waveform.to("cuda", torch.bfloat16), sample_rate


def encode_audio_to_latent(audio_data: str, target_frames: int) -> torch.Tensor:
    """Encode audio input to latent for conditioning."""
    audio_encoder, audio_processor = load_audio_encoder()

    waveform, sample_rate = decode_base64_audio(audio_data)
    print(f"  Audio input: {waveform.shape}, {sample_rate}Hz")

    mel = audio_processor.waveform_to_mel(waveform, waveform_sample_rate=sample_rate)
    print(f"  Mel spectrogram: {mel.shape}")

    with torch.no_grad():
        latent = audio_encoder(mel)
    print(f"  Audio latent (encoded): {latent.shape}")

    # Resize to match video frames
    if latent.shape[2] != target_frames:
        print(f"  Resizing audio latent from {latent.shape[2]} to {target_frames} frames")
        B, C, T, S = latent.shape
        latent = latent.permute(0, 1, 3, 2).reshape(B * C, S, T)
        latent = torch.nn.functional.interpolate(latent, size=target_frames, mode='linear', align_corners=False)
        latent = latent.reshape(B, C, S, -1).permute(0, 1, 3, 2)
        print(f"  Audio latent (resized): {latent.shape}")

    return latent


def decode_base64_image(image_data: str):
    """Decode base64 image to PIL Image."""
    from PIL import Image

    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def save_image_to_temp(image, width: int, height: int) -> str:
    """Save and resize image to temp file."""
    image = image.resize((width, height))
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    return temp_file.name


def encode_video_with_audio(video_tensor, audio_tensor, fps: int, output_path: str):
    """Encode video and audio to MP4."""
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        audio_path = os.path.join(tmpdir, "audio.wav") if audio_tensor is not None else None

        # Handle video frames
        if hasattr(video_tensor, '__iter__') and not isinstance(video_tensor, torch.Tensor):
            frames = []
            for chunk in video_tensor:
                if isinstance(chunk, torch.Tensor):
                    frames.append(chunk)
            if frames:
                video_tensor = torch.cat(frames, dim=0) if frames[0].dim() == 4 else torch.stack(frames, dim=0)

        if isinstance(video_tensor, torch.Tensor):
            video_np = video_tensor.cpu().numpy()
            if video_np.max() <= 1.0:
                video_np = (video_np * 255).astype(np.uint8)
            else:
                video_np = video_np.astype(np.uint8)

            if video_np.ndim == 5:
                video_np = video_np[0]

            from PIL import Image
            print(f"  Saving {len(video_np)} frames")
            for i, frame in enumerate(video_np):
                if frame.shape[0] in [1, 3, 4] and frame.shape[0] < frame.shape[1]:
                    frame = frame.transpose(1, 2, 0)
                if frame.shape[-1] == 1:
                    frame = frame.squeeze(-1)
                Image.fromarray(frame).save(frame_pattern % i)

        # Handle audio
        if audio_tensor is not None:
            audio_np = audio_tensor.cpu().numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
            if audio_np.ndim > 1:
                if audio_np.shape[0] in [1, 2] and audio_np.shape[0] < audio_np.shape[-1]:
                    audio_np = audio_np.T
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            wavfile.write(audio_path, AUDIO_SAMPLE_RATE, audio_int16)

        # Encode with ffmpeg
        if audio_path and os.path.exists(audio_path):
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(fps), "-i", frame_pattern,
                "-i", audio_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-crf", "18", "-shortest", output_path
            ], check=True, capture_output=True)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(fps), "-i", frame_pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path
            ], check=True, capture_output=True)


def generate_with_audio_conditioning(
    pipeline, prompt: str, negative_prompt: str, seed: int,
    height: int, width: int, num_frames: int, frame_rate: float,
    num_inference_steps: int, guidance_scale: float, images: list,
    audio_latent: torch.Tensor, audio_noise_scale: float = 0.15,
):
    """Generate video conditioned on audio input via cross-attention."""
    from ltx_core.components.diffusion_steps import EulerDiffusionStep
    from ltx_core.components.guiders import CFGGuider
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
    from ltx_core.model.video_vae import decode_video as vae_decode_video
    from ltx_core.text_encoders.gemma import encode_text
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
    from ltx_pipelines.utils.helpers import (
        cleanup_memory, euler_denoising_loop, guider_denoising_func,
        image_conditionings_by_replacing_latent, state_with_conditionings,
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(guidance_scale)

    text_encoder = pipeline.model_ledger.text_encoder()
    context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
    v_context_p, a_context_p = context_p
    v_context_n, a_context_n = context_n
    del text_encoder
    cleanup_memory()

    video_encoder = pipeline.model_ledger.video_encoder()
    transformer = pipeline.model_ledger.transformer()
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=pipeline.device)

    def denoising_loop(sigmas, video_state, audio_state, stepper):
        return euler_denoising_loop(
            sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper,
            denoise_fn=guider_denoising_func(cfg_guider, v_context_p, v_context_n, a_context_p, a_context_n, transformer=transformer),
        )

    output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
    conditionings = image_conditionings_by_replacing_latent(
        images=images, height=height, width=width,
        video_encoder=video_encoder, dtype=torch.bfloat16, device=pipeline.device,
    )

    # Video: full noise (standard generation)
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=pipeline.pipeline_components.video_latent_channels,
        scale_factors=pipeline.pipeline_components.video_scale_factors,
    )
    video_tools = VideoLatentTools(pipeline.pipeline_components.video_patchifier, video_latent_shape, output_shape.fps)
    video_state = video_tools.create_initial_state(pipeline.device, torch.bfloat16, initial_latent=None)
    video_state = state_with_conditionings(video_state, conditionings, video_tools)
    video_state = noiser(video_state, noise_scale=1.0)

    # Audio: low noise to preserve input (audio guides video via cross-attention)
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    audio_tools = AudioLatentTools(pipeline.pipeline_components.audio_patchifier, audio_latent_shape)
    audio_state = audio_tools.create_initial_state(pipeline.device, torch.bfloat16, initial_latent=audio_latent)
    audio_state = noiser(audio_state, noise_scale=audio_noise_scale)

    print(f"  Audio conditioning: noise_scale={audio_noise_scale}")

    video_state, audio_state = denoising_loop(sigmas, video_state, audio_state, stepper)

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    del transformer
    cleanup_memory()

    decoded_video = vae_decode_video(video_state.latent, pipeline.model_ledger.video_decoder(), generator=generator)
    decoded_audio = vae_decode_audio(audio_state.latent, pipeline.model_ledger.audio_decoder(), pipeline.model_ledger.vocoder())

    return decoded_video, decoded_audio


def generate_video(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    fps: int = 25,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 40,
    seed: Optional[int] = None,
    image: Optional[str] = None,
    image_cond_noise_scale: float = 0.15,
    include_audio: bool = True,
    audio: Optional[str] = None,
    audio_cond_noise_scale: float = 0.15,
) -> dict:
    """Generate video from text, optionally conditioned on image and/or audio."""
    pipeline = load_pipeline()

    # Ensure dimensions are divisible by 32
    width = (width // 32) * 32
    height = (height // 32) * 32

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    print(f"Generating video: {prompt[:50]}...")
    print(f"  Resolution: {width}x{height}, Frames: {num_frames}, Seed: {seed}")

    # Prepare image conditioning
    images: List[Tuple[str, int, float]] = []
    temp_image_path = None
    if image:
        print("  Using image conditioning")
        img = decode_base64_image(image)
        temp_image_path = save_image_to_temp(img, width, height)
        images = [(temp_image_path, 0, 1.0 - image_cond_noise_scale)]

    # Prepare audio conditioning
    audio_latent = None
    if audio:
        print("  Using audio conditioning")
        audio_latent = encode_audio_to_latent(audio, target_frames=num_frames)

    start = time.time()

    try:
        with torch.no_grad():
            if audio_latent is not None:
                # Audio-to-video: use custom function for cross-attention conditioning
                video_tensor, audio_tensor = generate_with_audio_conditioning(
                    pipeline=pipeline, prompt=prompt, negative_prompt=negative_prompt,
                    seed=seed, height=height, width=width, num_frames=num_frames,
                    frame_rate=float(fps), num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, images=images,
                    audio_latent=audio_latent, audio_noise_scale=audio_cond_noise_scale,
                )
            else:
                # Standard text/image-to-video: use pipeline directly
                video_tensor, audio_tensor = pipeline(
                    prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                    height=height, width=width, num_frames=num_frames,
                    frame_rate=float(fps), num_inference_steps=num_inference_steps,
                    cfg_guidance_scale=guidance_scale, images=images,
                )

        generation_time = time.time() - start
        print(f"Generation completed in {generation_time:.1f}s")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        encode_video_with_audio(video_tensor, audio_tensor if include_audio else None, fps, output_path)

        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "video": f"data:video/mp4;base64,{video_base64}",
            "duration_seconds": round(num_frames / fps, 2),
            "resolution": f"{width}x{height}",
            "fps": fps,
            "seed": seed,
            "generation_time_seconds": round(generation_time, 2),
            "mode": "audio-to-video" if audio else ("image-to-video" if image else "text-to-video"),
            "has_audio": include_audio,
        }
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.unlink(output_path)


def handler(event: dict) -> dict:
    """RunPod handler function."""
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        return generate_video(
            prompt=prompt,
            negative_prompt=input_data.get("negative_prompt", "blurry, low quality, distorted, glitchy, watermark"),
            width=input_data.get("width", 768),
            height=input_data.get("height", 512),
            num_frames=input_data.get("num_frames", 121),
            fps=input_data.get("fps", 25),
            guidance_scale=input_data.get("guidance_scale", 3.0),
            num_inference_steps=input_data.get("num_inference_steps", 40),
            seed=input_data.get("seed"),
            image=input_data.get("image"),
            image_cond_noise_scale=input_data.get("image_cond_noise_scale", 0.15),
            include_audio=input_data.get("include_audio", True),
            audio=input_data.get("audio"),
            audio_cond_noise_scale=input_data.get("audio_cond_noise_scale", 0.15),
        )
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
