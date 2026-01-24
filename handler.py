"""
RunPod LTX-2 Worker Handler
Following Wan2GP implementation pattern with two-stage distilled pipeline.
Uses DeepBeepMeep/LTX-2 model repository.
"""

import base64
import gc
import io
import json
import math
import os
import subprocess
import tempfile
import time
import types
from typing import Any, Iterator

import numpy as np
import runpod
import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from scipy.io import wavfile

# Set up files_locator paths before importing ltx modules
CACHE_DIR = os.environ.get("HF_HOME", "/runpod-volume/huggingface")

# Local imports
from shared.utils import files_locator as fl
fl.set_checkpoints_paths([CACHE_DIR, "ckpts", "."])

from ltx_core.conditioning import AudioConditionByLatent
from ltx_core.model.audio_vae import (
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    AudioProcessor,
    VocoderConfigurator,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.upsampler import LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)
from ltx_core.text_encoders.gemma import (
    GemmaTextEmbeddingsConnectorModelConfigurator,
    TEXT_EMBEDDING_PROJECTION_KEY_OPS,
    TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
    build_gemma_text_encoder,
)
from ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from ltx_core.types import AudioLatentShape, VideoPixelShape
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_NEGATIVE_PROMPT

# Configuration
MODEL_REPO = "DeepBeepMeep/LTX-2"
MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "distilled")  # "distilled" or "two_stage"

# Gemma text encoder folder
_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"

# Model files
MODEL_FILES = {
    "video_vae": "ltx-2-19b_vae.safetensors",
    "audio_vae": "ltx-2-19b_audio_vae.safetensors",
    "vocoder": "ltx-2-19b_vocoder.safetensors",
    "text_embedding_projection": "ltx-2-19b_text_embedding_projection.safetensors",
    "spatial_upsampler": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
}

# Transformer files based on variant
TRANSFORMER_FILES = {
    "distilled": "ltx-2-19b-distilled.safetensors",
    "two_stage": "ltx-2-19b-dev.safetensors",
}

# Embeddings connector based on variant
EMBEDDINGS_CONNECTORS = {
    "distilled": "ltx-2-19b-distilled_embeddings_connector.safetensors",
    "two_stage": "ltx-2-19b-dev_embeddings_connector.safetensors",
}

# Updated constants (from Wan2GP)
DEFAULT_FPS = 24.0
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_NUM_INFERENCE_STEPS = 40

# Global pipeline instance
pipeline = None
models = None


def cleanup_gpu_memory():
    """Force cleanup of GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def download_model_file(filename: str, subfolder: str = None) -> str:
    """Download a model file from HuggingFace."""
    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=filename,
        subfolder=subfolder,
        cache_dir=CACHE_DIR,
        local_dir_use_symlinks=False,
    )


def _normalize_config(config_value):
    """Normalize config from metadata."""
    if isinstance(config_value, dict):
        return config_value
    if isinstance(config_value, (bytes, bytearray, memoryview)):
        try:
            config_value = bytes(config_value).decode("utf-8")
        except Exception:
            return {}
    if isinstance(config_value, str):
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            return {}
    return {}


def _load_config_from_checkpoint(path: str) -> dict:
    """Load config from checkpoint metadata."""
    if isinstance(path, (list, tuple)):
        if not path:
            return {}
        path = path[0]
    if not path:
        return {}

    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        return _normalize_config(metadata.get("config"))


def _strip_model_prefix(key: str) -> str:
    """Strip model prefix from key."""
    if key.startswith("model."):
        return key[len("model."):]
    return key


def _apply_sd_ops(state_dict: dict, sd_ops):
    """Apply state dict operations."""
    if sd_ops is None:
        return {_strip_model_prefix(k): v for k, v in state_dict.items()}

    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        new_key = sd_ops.apply_to_key(key)
        if new_key is None:
            continue
        new_pairs = sd_ops.apply_to_key_value(new_key, value)
        for pair in new_pairs:
            new_sd[pair.new_key] = pair.new_value

    return new_sd


def _split_vae_state_dict(state_dict: dict, prefix: str):
    """Split VAE state dict."""
    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if key.startswith(prefix):
            key = key[len(prefix):]
        elif key.startswith(("encoder.", "decoder.", "per_channel_statistics.")):
            pass
        else:
            continue
        if key.startswith("per_channel_statistics."):
            suffix = key[len("per_channel_statistics."):]
            new_sd[f"encoder.per_channel_statistics.{suffix}"] = value.clone()
            new_sd[f"decoder.per_channel_statistics.{suffix}"] = value.clone()
        else:
            new_sd[key] = value
    return new_sd


class _VAEContainer(torch.nn.Module):
    """Container for VAE encoder and decoder."""
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def _load_component(model, path: str, sd_ops=None, dtype=torch.bfloat16, preprocess_sd=None):
    """Load model component from safetensors file."""
    sd = load_file(path)

    # Apply preprocessing
    if preprocess_sd is not None:
        sd = preprocess_sd(sd)
    elif sd_ops is not None:
        sd = _apply_sd_ops(sd, sd_ops)
    else:
        sd = {_strip_model_prefix(k): v for k, v in sd.items()}

    # Convert to target dtype
    sd = {k: v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in sd.items()}

    # Load into model
    model.load_state_dict(sd, strict=False, assign=True)
    model.eval()
    model.requires_grad_(False)
    return model


def init_models(variant: str = "distilled"):
    """Initialize all model components."""
    global models

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Initializing LTX-2 models with variant: {variant}")

    # Download and get paths
    print("Downloading model files...")
    transformer_file = TRANSFORMER_FILES.get(variant, TRANSFORMER_FILES["distilled"])
    transformer_path = download_model_file(transformer_file)
    print(f"  Transformer: {transformer_path}")

    embeddings_connector_file = EMBEDDINGS_CONNECTORS.get(variant, EMBEDDINGS_CONNECTORS["distilled"])
    embeddings_connector_path = download_model_file(embeddings_connector_file)
    print(f"  Embeddings connector: {embeddings_connector_path}")

    video_vae_path = download_model_file(MODEL_FILES["video_vae"])
    audio_vae_path = download_model_file(MODEL_FILES["audio_vae"])
    vocoder_path = download_model_file(MODEL_FILES["vocoder"])
    text_projection_path = download_model_file(MODEL_FILES["text_embedding_projection"])
    spatial_upsampler_path = download_model_file(MODEL_FILES["spatial_upsampler"])

    # Download Gemma files
    gemma_files = [
        "tokenizer.json",
        "tokenizer.model",
        "config_light.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]
    # The Gemma safetensors file is inside the subfolder with same name
    gemma_safetensors = download_model_file(f"{_GEMMA_FOLDER}/{_GEMMA_FOLDER}.safetensors")
    for gf in gemma_files:
        try:
            download_model_file(f"{_GEMMA_FOLDER}/{gf}")
        except Exception as e:
            print(f"  Warning: Could not download {gf}: {e}")

    print("Loading model components...")

    # Load configs
    base_config = _load_config_from_checkpoint(transformer_path)
    if not base_config:
        raise ValueError("Missing config in transformer checkpoint")

    def get_config(path):
        config = _load_config_from_checkpoint(path)
        return config or base_config

    # Load transformer
    print("  Loading transformer...")
    with init_empty_weights():
        velocity_model = LTXModelConfigurator.from_config(base_config)
    velocity_model = _load_component(velocity_model, transformer_path, LTXV_MODEL_COMFY_RENAMING_MAP, dtype)
    transformer = X0Model(velocity_model)
    transformer.to(device).eval().requires_grad_(False)

    # Load video VAE
    print("  Loading video VAE...")
    video_config = get_config(video_vae_path)
    with init_empty_weights():
        video_encoder = VideoEncoderConfigurator.from_config(video_config)
        video_decoder = VideoDecoderConfigurator.from_config(video_config)
        video_vae = _VAEContainer(video_encoder, video_decoder)
    video_vae = _load_component(
        video_vae, video_vae_path,
        preprocess_sd=lambda sd: _split_vae_state_dict(sd, "vae."),
        dtype=dtype
    )
    video_encoder = video_vae.encoder.to(device)
    video_decoder = video_vae.decoder.to(device)

    # Load audio VAE
    print("  Loading audio VAE...")
    audio_config = get_config(audio_vae_path)
    with init_empty_weights():
        audio_encoder = AudioEncoderConfigurator.from_config(audio_config)
        audio_decoder = AudioDecoderConfigurator.from_config(audio_config)
        audio_vae = _VAEContainer(audio_encoder, audio_decoder)
    audio_vae = _load_component(
        audio_vae, audio_vae_path,
        preprocess_sd=lambda sd: _split_vae_state_dict(sd, "audio_vae."),
        dtype=dtype
    )
    audio_encoder = audio_vae.encoder.to(device)
    audio_decoder = audio_vae.decoder.to(device)

    # Load vocoder
    print("  Loading vocoder...")
    vocoder_config = get_config(vocoder_path)
    with init_empty_weights():
        vocoder = VocoderConfigurator.from_config(vocoder_config)
    vocoder = _load_component(vocoder, vocoder_path, VOCODER_COMFY_KEYS_FILTER, dtype)
    vocoder.to(device)

    # Load text embedding projection
    print("  Loading text embedding projection...")
    text_projection_config = get_config(text_projection_path)
    with init_empty_weights():
        text_embedding_projection = GemmaFeaturesExtractorProjLinear.from_config(text_projection_config)
    text_embedding_projection = _load_component(
        text_embedding_projection, text_projection_path, TEXT_EMBEDDING_PROJECTION_KEY_OPS, dtype
    )
    text_embedding_projection.to(device)

    # Load text embeddings connector
    print("  Loading text embeddings connector...")
    text_connector_config = get_config(embeddings_connector_path)
    with init_empty_weights():
        text_embeddings_connector = GemmaTextEmbeddingsConnectorModelConfigurator.from_config(text_connector_config)
    text_embeddings_connector = _load_component(
        text_embeddings_connector, embeddings_connector_path, TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS, dtype
    )
    text_embeddings_connector.to(device)

    # Load Gemma text encoder
    print("  Loading Gemma text encoder...")
    text_encoder = build_gemma_text_encoder(gemma_safetensors, default_dtype=dtype)
    text_encoder.to(device).eval().requires_grad_(False)

    # Attach connectors to text encoder (required by pipeline)
    # Note: text_embeddings_connector is a wrapper containing video and audio connectors
    text_encoder.feature_extractor_linear = text_embedding_projection
    text_encoder.embeddings_connector = text_embeddings_connector.video_embeddings_connector
    text_encoder.audio_embeddings_connector = text_embeddings_connector.audio_embeddings_connector

    # Load spatial upsampler
    print("  Loading spatial upsampler...")
    upsampler_config = _load_config_from_checkpoint(spatial_upsampler_path)
    with init_empty_weights():
        spatial_upsampler = LatentUpsamplerConfigurator.from_config(upsampler_config)
    spatial_upsampler = _load_component(spatial_upsampler, spatial_upsampler_path, None, dtype)
    spatial_upsampler.to(device)

    print("All models loaded successfully")

    # Create models namespace
    models = types.SimpleNamespace(
        text_encoder=text_encoder,
        text_embedding_projection=text_embedding_projection,
        text_embeddings_connector=text_embeddings_connector,
        video_encoder=video_encoder,
        video_decoder=video_decoder,
        audio_encoder=audio_encoder,
        audio_decoder=audio_decoder,
        vocoder=vocoder,
        spatial_upsampler=spatial_upsampler,
        transformer=transformer,
    )

    return models


def init_pipeline(variant: str = "distilled"):
    """Initialize the pipeline with all models."""
    global pipeline, models

    if models is None:
        models = init_models(variant)

    device = torch.device("cuda")

    if variant == "distilled":
        pipeline = DistilledPipeline(
            device=device,
            models=models,
        )
    else:
        pipeline = TI2VidTwoStagesPipeline(
            device=device,
            stage_1_models=models,
            stage_2_models=models,
        )

    print(f"Pipeline initialized: {type(pipeline).__name__}")
    return pipeline


def _normalize_tiling_size(tile_size: int) -> int:
    """Normalize tiling size."""
    tile_size = int(tile_size)
    if tile_size <= 0:
        return 0
    tile_size = max(64, tile_size)
    if tile_size % 32 != 0:
        tile_size = int(math.ceil(tile_size / 32) * 32)
    return tile_size


def _normalize_temporal_tiling_size(tile_frames: int) -> int:
    """Normalize temporal tiling size."""
    tile_frames = int(tile_frames)
    if tile_frames <= 0:
        return 0
    tile_frames = max(16, tile_frames)
    if tile_frames % 8 != 0:
        tile_frames = int(math.ceil(tile_frames / 8) * 8)
    return tile_frames


def _normalize_temporal_overlap(overlap_frames: int, tile_frames: int) -> int:
    """Normalize temporal overlap."""
    overlap_frames = max(0, int(overlap_frames))
    if overlap_frames % 8 != 0:
        overlap_frames = int(round(overlap_frames / 8) * 8)
    overlap_frames = max(0, min(overlap_frames, max(0, tile_frames - 8)))
    return overlap_frames


def _build_tiling_config(tile_size: int | None, fps: float | None) -> TilingConfig | None:
    """Build tiling config for VAE."""
    spatial_config = None
    if tile_size is not None:
        tile_size = _normalize_tiling_size(tile_size)
        if tile_size > 0:
            overlap = max(0, tile_size // 4)
            overlap = int(math.floor(overlap / 32) * 32)
            if overlap >= tile_size:
                overlap = max(0, tile_size - 32)
            spatial_config = SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=overlap)

    temporal_config = None
    if fps is not None and fps > 0:
        tile_frames = _normalize_temporal_tiling_size(int(math.ceil(float(fps) * 5.0)))
        if tile_frames > 0:
            overlap_frames = int(round(tile_frames * 3 / 8))
            overlap_frames = _normalize_temporal_overlap(overlap_frames, tile_frames)
            temporal_config = TemporalTilingConfig(
                tile_size_in_frames=tile_frames,
                tile_overlap_in_frames=overlap_frames,
            )

    if spatial_config is None and temporal_config is None:
        return None
    return TilingConfig(spatial_config=spatial_config, temporal_config=temporal_config)


def _collect_video_chunks(video: Iterator[torch.Tensor] | torch.Tensor) -> torch.Tensor | None:
    """Collect video chunks into a single tensor."""
    if video is None:
        return None
    if torch.is_tensor(video):
        chunks = [video]
    else:
        chunks = []
        for chunk in video:
            if chunk is None:
                continue
            chunks.append(chunk if torch.is_tensor(chunk) else torch.tensor(chunk))
    if not chunks:
        return None
    frames = torch.cat(chunks, dim=0)
    return frames.permute(3, 0, 1, 2)


def encode_audio_to_latent(audio_data: bytes, target_frames: int, fps: float) -> torch.Tensor | None:
    """Encode audio to latent for conditioning."""
    global models

    if models is None:
        return None

    try:
        # Load audio from bytes
        audio_io = io.BytesIO(audio_data)
        sample_rate, waveform_np = wavfile.read(audio_io)

        # Convert to float tensor
        if waveform_np.dtype == np.int16:
            waveform_np = waveform_np.astype(np.float32) / 32768.0
        elif waveform_np.dtype == np.int32:
            waveform_np = waveform_np.astype(np.float32) / 2147483648.0
        elif waveform_np.dtype == np.uint8:
            waveform_np = (waveform_np.astype(np.float32) - 128) / 128.0

        waveform = torch.from_numpy(waveform_np.copy())
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.T.unsqueeze(0)

        # Match audio encoder channels
        target_channels = int(getattr(models.audio_encoder, "in_channels", waveform.shape[1]))
        if target_channels <= 0:
            target_channels = waveform.shape[1]
        if waveform.shape[1] != target_channels:
            if waveform.shape[1] == 1 and target_channels > 1:
                waveform = waveform.repeat(1, target_channels, 1)
            elif target_channels == 1:
                waveform = waveform.mean(dim=1, keepdim=True)

        # Process audio
        audio_processor = AudioProcessor(
            sample_rate=models.audio_encoder.sample_rate,
            mel_bins=models.audio_encoder.mel_bins,
            mel_hop_length=models.audio_encoder.mel_hop_length,
            n_fft=models.audio_encoder.n_fft,
        )
        waveform = waveform.to(device="cpu", dtype=torch.float32)
        audio_processor = audio_processor.to(waveform.device)
        mel = audio_processor.waveform_to_mel(waveform, sample_rate)

        # Encode audio
        device = torch.device("cuda")
        dtype = torch.bfloat16
        mel = mel.to(device=device, dtype=dtype)
        with torch.inference_mode():
            audio_latent = models.audio_encoder(mel)

        # Adjust audio latent shape
        audio_downsample = getattr(
            getattr(models.audio_encoder, "patchifier", None),
            "audio_latent_downsample_factor",
            4,
        )
        target_shape = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(
                batch=audio_latent.shape[0],
                frames=int(target_frames),
                width=1,
                height=1,
                fps=float(fps),
            ),
            channels=audio_latent.shape[1],
            mel_bins=audio_latent.shape[3],
            sample_rate=models.audio_encoder.sample_rate,
            hop_length=models.audio_encoder.mel_hop_length,
            audio_latent_downsample_factor=audio_downsample,
        )
        target_audio_frames = target_shape.frames
        if audio_latent.shape[2] < target_audio_frames:
            pad_frames = target_audio_frames - audio_latent.shape[2]
            pad = torch.zeros(
                (audio_latent.shape[0], audio_latent.shape[1], pad_frames, audio_latent.shape[3]),
                device=audio_latent.device,
                dtype=audio_latent.dtype,
            )
            audio_latent = torch.cat([audio_latent, pad], dim=2)
        elif audio_latent.shape[2] > target_audio_frames:
            audio_latent = audio_latent[:, :, :target_audio_frames, :]

        audio_latent = audio_latent.to(device=device, dtype=dtype)
        return audio_latent

    except Exception as e:
        print(f"Warning: Failed to process audio: {e}")
        return None


def generate_video(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    num_frames: int = 121,
    height: int = 512,
    width: int = 768,
    fps: float = DEFAULT_FPS,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int = 0,
    audio_data: bytes | None = None,
    audio_strength: float = 1.0,
    tile_size: int | None = None,
) -> dict[str, Any]:
    """Generate video from text prompt."""
    global pipeline, models

    if pipeline is None:
        init_pipeline(MODEL_VARIANT)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Align dimensions to 64
    target_height = int(height)
    target_width = int(width)
    if target_height % 64 != 0:
        target_height = int(math.ceil(target_height / 64) * 64)
    if target_width % 64 != 0:
        target_width = int(math.ceil(target_width / 64) * 64)

    # Build tiling config
    tiling_config = _build_tiling_config(tile_size, fps)

    # Process audio conditioning if provided
    audio_conditionings = None
    if audio_data is not None and audio_strength > 0.0:
        audio_latent = encode_audio_to_latent(audio_data, num_frames, fps)
        if audio_latent is not None:
            audio_conditionings = [AudioConditionByLatent(audio_latent, audio_strength)]

    # Generate video
    print(f"Generating video: {prompt[:60]}...")
    print(f"  Resolution: {target_width}x{target_height}, Frames: {num_frames}, Seed: {seed}")
    start_time = time.time()

    if isinstance(pipeline, TI2VidTwoStagesPipeline):
        neg_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT
        output = pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=int(seed),
            height=target_height,
            width=target_width,
            num_frames=int(num_frames),
            frame_rate=float(fps),
            num_inference_steps=int(num_inference_steps),
            cfg_guidance_scale=float(guidance_scale),
            images=[],
            tiling_config=tiling_config,
            enhance_prompt=False,
            audio_conditionings=audio_conditionings,
        )
    else:
        output = pipeline(
            prompt=prompt,
            seed=int(seed),
            height=target_height,
            width=target_width,
            num_frames=int(num_frames),
            frame_rate=float(fps),
            images=[],
            tiling_config=tiling_config,
            enhance_prompt=False,
            audio_conditionings=audio_conditionings,
        )

    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.1f}s")

    # Process output
    if isinstance(output, tuple) and len(output) >= 2:
        video, audio = output[0], output[1]
    else:
        video, audio = output, None

    if video is None:
        return {"error": "Video generation failed"}

    video_tensor = _collect_video_chunks(video)
    if video_tensor is None:
        return {"error": "Failed to collect video chunks"}

    # Trim to requested size
    video_tensor = video_tensor[:, :num_frames, :height, :width]

    # Convert to numpy
    video_np = video_tensor.float().cpu().numpy()

    # Process audio
    audio_np = None
    if audio is not None:
        audio_np = audio.detach().float().cpu().numpy()
        if audio_np.ndim == 2:
            if audio_np.shape[0] in (1, 2) and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T

    return {
        "video": video_np,
        "audio": audio_np,
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
        "fps": fps,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "generation_time": generation_time,
        "seed": seed,
    }


def video_to_base64(video_np: np.ndarray, fps: float = 24.0, audio_np: np.ndarray | None = None, audio_sr: int = 24000) -> str:
    """Convert video numpy array to base64 encoded MP4."""
    # Normalize video to uint8
    if video_np.dtype != np.uint8:
        video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    # video_np shape: (C, T, H, W) -> (T, H, W, C)
    if video_np.shape[0] in (1, 3, 4):
        video_np = video_np.transpose(1, 2, 3, 0)

    num_frames, height, width, channels = video_np.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "output.mp4")
        frame_pattern = os.path.join(tmpdir, "frame_%04d.png")

        # Save frames
        from PIL import Image
        for i, frame in enumerate(video_np):
            if channels == 1:
                frame = frame.squeeze(-1)
            Image.fromarray(frame).save(frame_pattern % i)

        # Build ffmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
        ]

        # Add audio if available
        audio_file = None
        if audio_np is not None:
            audio_file = os.path.join(tmpdir, "audio.wav")
            audio_int16 = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_int16 * 32767).astype(np.int16)
            wavfile.write(audio_file, audio_sr, audio_int16)
            cmd.extend(["-i", audio_file])

        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
        ])

        if audio_np is not None:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

        cmd.append(video_path)

        # Run ffmpeg
        subprocess.run(cmd, check=True, capture_output=True)

        # Read and encode
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def handler(event: dict) -> dict:
    """RunPod handler function."""
    try:
        job_input = event.get("input", {})

        # Extract parameters
        prompt = job_input.get("prompt", "")
        if not prompt:
            return {"error": "Prompt is required"}

        negative_prompt = job_input.get("negative_prompt", "blurry, low quality, distorted, glitchy, watermark")
        num_frames = job_input.get("num_frames", 121)
        height = job_input.get("height", 512)
        width = job_input.get("width", 768)
        fps = job_input.get("fps", DEFAULT_FPS)
        num_inference_steps = job_input.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
        guidance_scale = job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        seed = job_input.get("seed", 0)
        tile_size = job_input.get("tile_size")
        include_audio = job_input.get("include_audio", True)

        # Audio conditioning
        audio_base64 = job_input.get("audio")
        audio_data = None
        if audio_base64:
            if audio_base64.startswith("data:"):
                audio_base64 = audio_base64.split(",", 1)[1]
            audio_data = base64.b64decode(audio_base64)
        audio_strength = job_input.get("audio_strength", 1.0)

        # Generate video
        result = generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            audio_data=audio_data,
            audio_strength=audio_strength,
            tile_size=tile_size,
        )

        if "error" in result:
            return result

        # Encode video to base64
        video_base64 = video_to_base64(
            result["video"],
            fps=result["fps"],
            audio_np=result.get("audio") if include_audio else None,
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
            "has_audio": include_audio and result.get("audio") is not None,
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# Initialize on startup
print("=" * 60)
print(f"LTX-2 RunPod Worker - Variant: {MODEL_VARIANT}")
print(f"Model Repo: {MODEL_REPO}")
print(f"Using FPS: {DEFAULT_FPS}, Guidance: {DEFAULT_GUIDANCE_SCALE}")
print("=" * 60)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
