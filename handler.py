"""
RunPod Serverless Handler for LTX-2 Video Generation

API for generating videos with LTX-2 19B model.
Supports text-to-video, image-to-video, and audio-to-video generation.
Generates synchronized audio alongside video.

Audio INPUT is supported for conditioning video generation.
"""

import base64
import io
import os
import subprocess
import tempfile
import time
from typing import Optional, List, Tuple

import runpod
import torch

# Cache directory for model files
CACHE_DIR = os.environ.get("HF_HOME", "/runpod-volume/huggingface")
MODEL_ID = "Lightricks/LTX-2"
AUDIO_SAMPLE_RATE = 24000

# Global model cache
PIPELINE = None


class CachingModelLedger:
    """Wrapper around ModelLedger that caches model instances to prevent reloading."""

    def __init__(self, original_ledger):
        self._ledger = original_ledger
        self._cache = {}

    def _get_cached(self, name, builder_fn):
        if name not in self._cache:
            print(f"  Loading and caching {name}...")
            self._cache[name] = builder_fn()
        return self._cache[name]

    def text_encoder(self):
        return self._get_cached("text_encoder", self._ledger.text_encoder)

    def transformer(self):
        return self._get_cached("transformer", self._ledger.transformer)

    def video_encoder(self):
        return self._get_cached("video_encoder", self._ledger.video_encoder)

    def video_decoder(self):
        return self._get_cached("video_decoder", self._ledger.video_decoder)

    def audio_decoder(self):
        return self._get_cached("audio_decoder", self._ledger.audio_decoder)

    def vocoder(self):
        return self._get_cached("vocoder", self._ledger.vocoder)

    def spatial_upsampler(self):
        return self._get_cached("spatial_upsampler", self._ledger.spatial_upsampler)

    # Forward other attributes to original ledger
    def __getattr__(self, name):
        return getattr(self._ledger, name)


def download_model_files():
    """Download model files from HuggingFace."""
    from huggingface_hub import hf_hub_download, snapshot_download

    print("Downloading LTX-2 model files...")

    # Download distilled FP8 checkpoint (faster inference, same quality)
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="ltx-2-19b-distilled-fp8.safetensors",
        cache_dir=CACHE_DIR,
    )

    # Download text encoder (Gemma) and tokenizer
    model_root = snapshot_download(
        repo_id=MODEL_ID,
        allow_patterns=["text_encoder/*", "tokenizer/*"],
        cache_dir=CACHE_DIR,
    )
    gemma_root = model_root

    return checkpoint_path, gemma_root


def load_pipeline():
    """Load LTX-2 pipeline with model caching to prevent OOM."""
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    print("Loading LTX-2 pipeline...")
    start = time.time()

    # Download model files
    checkpoint_path, gemma_root = download_model_files()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Gemma root: {gemma_root}")

    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=checkpoint_path,
        gemma_root=gemma_root,
        loras=[],
        device="cuda",
        fp8transformer=True,  # Use FP8 to reduce memory usage
    )

    # Don't cache models - let pipeline manage memory by loading/unloading
    # The 19B model requires this approach to fit in GPU memory

    PIPELINE = pipeline
    print(f"Pipeline initialized in {time.time() - start:.1f}s")
    return PIPELINE


# Global audio encoder cache
AUDIO_ENCODER = None
AUDIO_PROCESSOR = None


def load_audio_encoder():
    """Load audio encoder for audio-to-video conditioning."""
    global AUDIO_ENCODER, AUDIO_PROCESSOR

    if AUDIO_ENCODER is not None:
        return AUDIO_ENCODER, AUDIO_PROCESSOR

    print("Loading audio encoder...")
    checkpoint_path, _ = download_model_files()

    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.audio_vae import (
        AudioEncoderConfigurator,
        AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        AudioProcessor,
    )

    builder = Builder(
        model_path=checkpoint_path,
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    )
    AUDIO_ENCODER = builder.build(device='cuda', dtype=torch.bfloat16)
    AUDIO_ENCODER.eval()

    AUDIO_PROCESSOR = AudioProcessor(
        sample_rate=16000,
        mel_bins=64,
        mel_hop_length=160,
        n_fft=1024,
    ).to('cuda')

    print("Audio encoder loaded")
    return AUDIO_ENCODER, AUDIO_PROCESSOR


def decode_base64_audio(audio_data: str) -> torch.Tensor:
    """Decode base64 audio to waveform tensor."""
    import scipy.io.wavfile as wavfile
    import numpy as np

    if audio_data.startswith("data:"):
        audio_data = audio_data.split(",", 1)[1]

    audio_bytes = base64.b64decode(audio_data)

    # Read WAV file
    with io.BytesIO(audio_bytes) as f:
        sample_rate, waveform = wavfile.read(f)

    # Convert to float tensor
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    elif waveform.dtype == np.uint8:
        waveform = (waveform.astype(np.float32) - 128) / 128.0

    # Convert to tensor [batch, channels, samples]
    waveform = torch.from_numpy(waveform)
    if waveform.ndim == 1:
        # Mono - duplicate to stereo
        waveform = waveform.unsqueeze(0).repeat(2, 1)
    else:
        # Stereo - transpose to [channels, samples]
        waveform = waveform.T

    waveform = waveform.unsqueeze(0)  # Add batch dim
    return waveform.to('cuda', torch.bfloat16), sample_rate


def encode_audio_to_latent(audio_data: str, target_frames: int) -> torch.Tensor:
    """Encode audio input to latent representation for conditioning.

    Args:
        audio_data: Base64-encoded WAV audio
        target_frames: Number of video frames to match (determines audio latent temporal dimension)
    """
    audio_encoder, audio_processor = load_audio_encoder()

    waveform, sample_rate = decode_base64_audio(audio_data)
    print(f"  Audio input: {waveform.shape}, {sample_rate}Hz")

    # Convert to mel spectrogram
    mel = audio_processor.waveform_to_mel(waveform, waveform_sample_rate=sample_rate)
    print(f"  Mel spectrogram: {mel.shape}")

    # Encode to latent
    with torch.no_grad():
        latent = audio_encoder(mel)
    print(f"  Audio latent (encoded): {latent.shape}")

    # Resize audio latent temporal dimension to match video frames
    # Latent shape is [batch, channels, temporal, spectral]
    if latent.shape[2] != target_frames:
        print(f"  Resizing audio latent from {latent.shape[2]} to {target_frames} frames")
        B, C, T, S = latent.shape
        # Reshape to [B*C, S, T] for 3D interpolation, then back
        latent = latent.permute(0, 1, 3, 2).reshape(B * C, S, T)  # [B*C, S, T]
        latent = torch.nn.functional.interpolate(
            latent,
            size=target_frames,
            mode='linear',
            align_corners=False,
        )  # [B*C, S, target_frames]
        latent = latent.reshape(B, C, S, -1).permute(0, 1, 3, 2)  # Back to [B, C, T, S]
        print(f"  Audio latent (resized): {latent.shape}")

    return latent


def decode_base64_image(image_data: str):
    """Decode base64 image to PIL Image."""
    from PIL import Image

    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def save_image_to_temp(image, width, height):
    """Save PIL image to temporary file."""
    image = image.resize((width, height))
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    return tmp.name


def _process_frame(frame):
    """Process a single frame to uint8 HWC format for PIL."""
    import numpy as np

    # Normalize to 0-255 range
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Handle channel dimension if in CHW format
    if frame.ndim == 3 and frame.shape[0] in [1, 3, 4] and frame.shape[2] not in [1, 3, 4]:
        frame = frame.transpose(1, 2, 0)

    # Squeeze single channel dimension
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame.squeeze(-1)

    return frame


def encode_video_with_audio(video_generator, audio_tensor, fps, output_path):
    """Encode video frames and audio to MP4 file."""
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")
        audio_path = os.path.join(tmpdir, "audio.wav")

        # Consume generator and save video frames
        # Each yielded tensor has shape [batch, height, width, channels] or [height, width, channels]
        frame_idx = 0
        for chunk in video_generator:
            if torch.is_tensor(chunk):
                chunk = chunk.cpu().numpy()

            # Handle batch dimension - chunk may be [B, H, W, C] or [H, W, C]
            if chunk.ndim == 4:
                # Multiple frames in this chunk
                for frame in chunk:
                    frame = _process_frame(frame)
                    Image.fromarray(frame).save(frame_pattern % frame_idx)
                    frame_idx += 1
            else:
                # Single frame [H, W, C]
                frame = _process_frame(chunk)
                Image.fromarray(frame).save(frame_pattern % frame_idx)
                frame_idx += 1

        print(f"  Saved {frame_idx} frames")

        # Save audio
        if audio_tensor is not None:
            import scipy.io.wavfile as wavfile
            if torch.is_tensor(audio_tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor

            # Audio tensor is [channels, samples] - transpose to [samples, channels] for wavfile
            if audio_np.ndim == 2 and audio_np.shape[0] in [1, 2]:
                audio_np = audio_np.T  # [samples, channels]

            # Normalize to int16 range
            if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
                audio_np = (audio_np * 32767).astype(np.int16)
            else:
                audio_np = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)

            wavfile.write(audio_path, AUDIO_SAMPLE_RATE, audio_np)

            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-i", audio_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-shortest",
                output_path
            ], check=True, capture_output=True)
        else:
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                output_path
            ], check=True, capture_output=True)


def generate_with_audio_conditioning(
    pipeline,
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    cfg_guidance_scale: float,
    images: list,
    audio_latent: torch.Tensor,
    audio_noise_scale: float = 0.15,
):
    """Generate video with audio conditioning using internal pipeline functions.

    The key insight: audio and video are denoised JOINTLY with cross-attention between them.
    - audio_to_video_attn: Audio latent influences video generation
    - video_to_audio_attn: Video latent influences audio generation

    By providing an initial audio latent with LOW noise_scale, we preserve the input audio
    information, which then guides video generation through cross-attention.

    Args:
        audio_latent: Encoded audio latent from AudioEncoder
        audio_noise_scale: How much noise to add to audio (0=exact audio, 1=ignore audio).
                          Lower values = stronger audio conditioning. Default 0.15.
    """
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
        assert_resolution,
        cleanup_memory,
        euler_denoising_loop,
        guider_denoising_func,
        image_conditionings_by_replacing_latent,
        state_with_conditionings,
    )

    assert_resolution(height=height, width=width, is_two_stage=False)

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    cfg_guider = CFGGuider(cfg_guidance_scale)
    dtype = torch.bfloat16

    # Encode text
    text_encoder = pipeline.model_ledger.text_encoder()
    context_p, context_n = encode_text(text_encoder, prompts=[prompt, negative_prompt])
    v_context_p, a_context_p = context_p
    v_context_n, a_context_n = context_n

    torch.cuda.synchronize()
    del text_encoder
    cleanup_memory()

    # Prepare video encoder and transformer
    video_encoder = pipeline.model_ledger.video_encoder()
    transformer = pipeline.model_ledger.transformer()
    sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=pipeline.device)

    def denoising_loop(sigmas, video_state, audio_state, stepper):
        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=guider_denoising_func(
                cfg_guider,
                v_context_p,
                v_context_n,
                a_context_p,
                a_context_n,
                transformer=transformer,
            ),
        )

    output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
    conditionings = image_conditionings_by_replacing_latent(
        images=images,
        height=output_shape.height,
        width=output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=pipeline.device,
    )

    # Initialize VIDEO state with full noise (noise_scale=1.0) for text-to-video
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=pipeline.pipeline_components.video_latent_channels,
        scale_factors=pipeline.pipeline_components.video_scale_factors,
    )
    video_tools = VideoLatentTools(
        pipeline.pipeline_components.video_patchifier, video_latent_shape, output_shape.fps
    )
    video_state = video_tools.create_initial_state(pipeline.device, dtype, initial_latent=None)
    video_state = state_with_conditionings(video_state, conditionings, video_tools)
    video_state = noiser(video_state, noise_scale=1.0)  # Full noise for video

    # Initialize AUDIO state with LOW noise to preserve input audio for conditioning
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    audio_tools = AudioLatentTools(pipeline.pipeline_components.audio_patchifier, audio_latent_shape)
    audio_state = audio_tools.create_initial_state(pipeline.device, dtype, initial_latent=audio_latent)
    audio_state = noiser(audio_state, noise_scale=audio_noise_scale)  # Low noise preserves audio

    print(f"  Audio conditioning: noise_scale={audio_noise_scale} (lower=stronger conditioning)")

    # Joint denoising - audio influences video through cross-attention
    video_state, audio_state = denoising_loop(sigmas, video_state, audio_state, stepper)

    # Unpatchify outputs
    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    torch.cuda.synchronize()
    del transformer
    cleanup_memory()

    # Decode outputs
    decoded_video = vae_decode_video(video_state.latent, pipeline.model_ledger.video_decoder(), generator=generator)
    decoded_audio = vae_decode_audio(
        audio_state.latent, pipeline.model_ledger.audio_decoder(), pipeline.model_ledger.vocoder()
    )

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
    """Generate a video from text prompt, optionally conditioned on an image and/or audio."""

    pipeline = load_pipeline()

    # Validate dimensions (must be divisible by 32)
    if width % 32 != 0:
        width = (width // 32) * 32
        print(f"  Adjusted width to {width} (must be divisible by 32)")
    if height % 32 != 0:
        height = (height // 32) * 32
        print(f"  Adjusted height to {height} (must be divisible by 32)")

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()

    print(f"Generating video: {prompt[:50]}...")
    print(f"  Resolution: {width}x{height}, Frames: {num_frames}, Seed: {seed}")

    images: List[Tuple[str, int, float]] = []
    temp_image_path = None
    if image:
        print("  Using image conditioning (image-to-video mode)")
        conditioning_image = decode_base64_image(image)
        temp_image_path = save_image_to_temp(conditioning_image, width, height)
        images = [(temp_image_path, 0, 1.0 - image_cond_noise_scale)]

    # Encode audio input if provided
    audio_latent = None
    if audio:
        print("  Using audio conditioning (audio-to-video mode)")
        audio_latent = encode_audio_to_latent(audio, target_frames=num_frames)

    start = time.time()

    try:
        # Use no_grad to prevent gradient memory allocation (critical for 19B model)
        with torch.no_grad():
            if audio_latent is not None:
                # Use custom generation with audio conditioning
                video_tensor, audio_tensor = generate_with_audio_conditioning(
                    pipeline=pipeline,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(fps),
                    num_inference_steps=num_inference_steps,
                    cfg_guidance_scale=guidance_scale,
                    images=images if images else [],
                    audio_latent=audio_latent,
                    audio_noise_scale=audio_cond_noise_scale,
                )
            else:
                video_tensor, audio_tensor = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(fps),
                    num_inference_steps=num_inference_steps,
                    cfg_guidance_scale=guidance_scale,
                    images=images if images else [],
                )

        generation_time = time.time() - start
        print(f"Generation completed in {generation_time:.1f}s")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_output:
            output_path = tmp_output.name

        encode_video_with_audio(
            video_tensor,
            audio_tensor if include_audio else None,
            fps,
            output_path
        )

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        duration = num_frames / fps

        return {
            "video": f"data:video/mp4;base64,{video_base64}",
            "duration_seconds": round(duration, 2),
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

        result = generate_video(
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

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
