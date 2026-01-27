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
from PIL import Image
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
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
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

# IC-LoRA control model files (downloaded lazily on first use)
IC_LORA_FILES = {
    "depth": "ltx-2-19b-ic-lora-depth-control.safetensors",
    "canny": "ltx-2-19b-ic-lora-canny-control.safetensors",
    "pose": "ltx-2-19b-ic-lora-pose-control.safetensors",
}

# Distilled LoRA for keyframe interpolation stage 2
DISTILLED_LORA_FILE = "ltx-2-19b-distilled-lora-384.safetensors"

# Updated constants (from Wan2GP)
DEFAULT_FPS = 24.0
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_NUM_INFERENCE_STEPS = 40

# FP8 quantization mode
FP8_MODE = os.environ.get("FP8_MODE", "false").lower() in ("true", "1", "yes")

# Global pipeline manager
pipeline_manager = None
# Keep backward-compat globals (used by encode_audio_to_latent)
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


# Global model paths dict (populated during init_models)
_model_paths: dict[str, str] = {}


def init_models(variant: str = "distilled"):
    """Initialize all model components."""
    global models, _model_paths

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

    # Store model file paths for lazy pipeline construction
    _model_paths.update({
        "transformer": transformer_path,
        "spatial_upsampler": spatial_upsampler_path,
        "gemma_root": gemma_safetensors,
        "video_vae": video_vae_path,
        "audio_vae": audio_vae_path,
        "vocoder": vocoder_path,
        "text_projection": text_projection_path,
        "embeddings_connector": embeddings_connector_path,
    })

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
    """Initialize the pipeline manager with all models."""
    global pipeline, models, pipeline_manager

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

    # Create pipeline manager wrapping the default pipeline
    pipeline_manager = PipelineManager(
        variant=variant,
        default_pipeline=pipeline,
        models=models,
        model_paths=_model_paths,
    )

    return pipeline


class PipelineManager:
    """Manages pipeline lifecycle and lazy switching between pipeline types."""

    def __init__(self, variant: str, default_pipeline, models, model_paths: dict):
        self.variant = variant
        self.default_pipeline = default_pipeline
        self.models = models
        self.model_paths = model_paths
        self.current_pipeline = default_pipeline
        self.current_type = "default"
        self._ic_lora_paths: dict[str, str] = {}
        self._distilled_lora_path: str | None = None
        self._dev_transformer_path: str | None = None

    def _ensure_dev_transformer(self) -> str:
        """Download the dev transformer if not already available."""
        if self._dev_transformer_path is not None:
            return self._dev_transformer_path
        # If current variant is two_stage, the transformer IS the dev model
        if self.variant == "two_stage":
            self._dev_transformer_path = self.model_paths["transformer"]
            return self._dev_transformer_path
        # Otherwise download the dev model
        print("Downloading dev transformer for IC-LoRA/keyframe pipeline...")
        self._dev_transformer_path = download_model_file(TRANSFORMER_FILES["two_stage"])
        print(f"  Dev transformer: {self._dev_transformer_path}")
        return self._dev_transformer_path

    def _ensure_ic_lora(self, control_type: str) -> str:
        """Download IC-LoRA file for the given control type."""
        if control_type in self._ic_lora_paths:
            return self._ic_lora_paths[control_type]
        filename = IC_LORA_FILES.get(control_type)
        if filename is None:
            raise ValueError(f"Unknown control type: {control_type}. Must be one of: {list(IC_LORA_FILES.keys())}")
        print(f"Downloading IC-LoRA model for {control_type}...")
        path = download_model_file(filename)
        self._ic_lora_paths[control_type] = path
        print(f"  IC-LoRA {control_type}: {path}")
        return path

    def _ensure_distilled_lora(self) -> str:
        """Download the distilled LoRA for keyframe interpolation."""
        if self._distilled_lora_path is not None:
            return self._distilled_lora_path
        print("Downloading distilled LoRA for keyframe pipeline...")
        self._distilled_lora_path = download_model_file(DISTILLED_LORA_FILE)
        print(f"  Distilled LoRA: {self._distilled_lora_path}")
        return self._distilled_lora_path

    def get_pipeline(self, mode: str, fp8: bool = False, control_type: str | None = None,
                     lora_entries: list | None = None):
        """Get the appropriate pipeline for the given mode.

        Returns the pipeline instance. Switches pipelines lazily as needed.
        """
        device = torch.device("cuda")
        use_fp8 = fp8 or FP8_MODE

        if mode == "controlnet":
            if not control_type:
                raise ValueError("control_type is required for controlnet mode")
            ic_lora_path = self._ensure_ic_lora(control_type)
            dev_path = self._ensure_dev_transformer()
            loras = [LoraPathStrengthAndSDOps(ic_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
            if lora_entries:
                loras.extend(lora_entries)
            pipe_key = f"controlnet_{control_type}"
            if self.current_type != pipe_key:
                print(f"Switching to ICLoraPipeline ({control_type})...")
                self._release_current()
                self.current_pipeline = ICLoraPipeline(
                    checkpoint_path=dev_path,
                    spatial_upsampler_path=self.model_paths["spatial_upsampler"],
                    gemma_root=self.model_paths["gemma_root"],
                    loras=loras,
                    device=device,
                    fp8transformer=use_fp8,
                )
                self.current_type = pipe_key
            return self.current_pipeline

        if mode == "keyframe_interpolation":
            dev_path = self._ensure_dev_transformer()
            distilled_lora_path = self._ensure_distilled_lora()
            distilled_lora = [LoraPathStrengthAndSDOps(distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
            user_loras = list(lora_entries) if lora_entries else []
            pipe_key = "keyframe_interpolation"
            if self.current_type != pipe_key:
                print("Switching to KeyframeInterpolationPipeline...")
                self._release_current()
                self.current_pipeline = KeyframeInterpolationPipeline(
                    checkpoint_path=dev_path,
                    distilled_lora=distilled_lora,
                    spatial_upsampler_path=self.model_paths["spatial_upsampler"],
                    gemma_root=self.model_paths["gemma_root"],
                    loras=user_loras,
                    device=device,
                    fp8transformer=use_fp8,
                )
                self.current_type = pipe_key
            return self.current_pipeline

        # Default modes: text_to_video, image_to_video, inpainting, video_extension, video_to_audio
        if lora_entries:
            # LoRA requires ModelLedger path — create a temporary pipeline
            print("Creating temporary pipeline with LoRA support...")
            if self.variant == "distilled":
                return DistilledPipeline(
                    checkpoint_path=self.model_paths["transformer"],
                    spatial_upsampler_path=self.model_paths["spatial_upsampler"],
                    gemma_root=self.model_paths["gemma_root"],
                    loras=list(lora_entries),
                    device=device,
                    fp8transformer=use_fp8,
                )
            else:
                return TI2VidTwoStagesPipeline(
                    checkpoint_path=self.model_paths["transformer"],
                    spatial_upsampler_path=self.model_paths["spatial_upsampler"],
                    gemma_root=self.model_paths["gemma_root"],
                    loras=list(lora_entries),
                    device=device,
                    fp8transformer=use_fp8,
                )

        # Return the default (pre-loaded) pipeline
        if self.current_type != "default":
            print("Switching back to default pipeline...")
            self._release_current()
            self.current_pipeline = self.default_pipeline
            self.current_type = "default"
        return self.default_pipeline

    def _release_current(self):
        """Release the current pipeline to free GPU memory."""
        if self.current_pipeline is not None and self.current_pipeline is not self.default_pipeline:
            del self.current_pipeline
            self.current_pipeline = None
            cleanup_gpu_memory()


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


def _to_latent_index(frame_idx: int, stride: int) -> int:
    """Convert pixel frame index to latent index."""
    return int(frame_idx) // int(stride)


def _decode_base64_image(image_data: str) -> Image.Image | None:
    """Decode a base64 encoded image to PIL Image."""
    if not image_data:
        return None
    try:
        # Handle data URL format
        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print(f"Warning: Failed to decode image: {e}")
        return None


def _decode_base64_video(data: str) -> str | None:
    """Decode a base64 encoded video to a temp file path.

    Pipelines expect file paths for video inputs. Returns the path to a
    temporary MP4 file. Caller is responsible for cleanup.
    """
    if not data:
        return None
    try:
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        video_bytes = base64.b64decode(data)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(video_bytes)
        tmp.close()
        return tmp.name
    except Exception as e:
        print(f"Warning: Failed to decode video: {e}")
        return None


def _decode_base64_mask(data: str, num_frames: int | None = None) -> torch.Tensor | None:
    """Decode a base64 mask image to a tensor.

    The mask image is interpreted as a single-frame spatial mask:
    white (255) = generate, black (0) = preserve. Returns a float tensor
    in [0, 1] range.
    """
    if not data:
        return None
    try:
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        mask_bytes = base64.b64decode(data)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_np = np.array(mask_img, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)
        if num_frames is not None and num_frames > 1:
            # Expand to temporal dimension: (T, H, W)
            mask_tensor = mask_tensor.unsqueeze(0).expand(num_frames, -1, -1)
        return mask_tensor
    except Exception as e:
        print(f"Warning: Failed to decode mask: {e}")
        return None


def _decode_keyframes(keyframes_list: list[dict]) -> list[tuple]:
    """Decode a list of keyframe dicts with base64 images.

    Each keyframe dict should have: {image: base64, frame_index: int, strength: float}
    Returns list of (PIL.Image, latent_frame_index, strength) tuples.
    """
    latent_stride = 8
    result = []
    for kf in keyframes_list:
        img = _decode_base64_image(kf.get("image", ""))
        if img is None:
            print(f"Warning: Skipping keyframe with invalid image")
            continue
        frame_index = int(kf.get("frame_index", 0))
        strength = float(kf.get("strength", 1.0))
        latent_idx = _to_latent_index(frame_index, latent_stride)
        result.append((img, latent_idx, strength))
    return result


def _resolve_lora_entries(loras_list: list[dict] | None) -> list[LoraPathStrengthAndSDOps] | None:
    """Resolve a list of LoRA dicts to LoraPathStrengthAndSDOps entries.

    Each dict should have: {path: str, strength: float}
    The path can be a local file path or a HuggingFace repo_id/filename.
    """
    if not loras_list:
        return None
    entries = []
    for lora in loras_list:
        path = lora.get("path", "")
        strength = float(lora.get("strength", 1.0))
        if not path:
            continue
        # If path contains '/' and doesn't exist locally, try HuggingFace download
        if "/" in path and not os.path.isfile(path):
            try:
                parts = path.split("/", 1)
                if len(parts) == 2:
                    path = hf_hub_download(repo_id=parts[0] + "/" + parts[1].split("/")[0],
                                           filename="/".join(parts[1].split("/")[1:]),
                                           cache_dir=CACHE_DIR)
                else:
                    path = download_model_file(path)
            except Exception:
                # Try as a direct model repo file
                try:
                    path = download_model_file(path)
                except Exception as e:
                    print(f"Warning: Could not resolve LoRA path '{path}': {e}")
                    continue
        entries.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    return entries if entries else None


def _align_frames(num_frames: int) -> int:
    """Align frame count to 8k+1 format expected by the video VAE.

    The video VAE temporal compression uses a stride of 8, so the total
    frame count should be 8k+1 for proper encoding.
    """
    if num_frames <= 1:
        return 1
    # Round up to nearest 8k+1
    k = math.ceil((num_frames - 1) / 8)
    return k * 8 + 1


def _extract_frames_from_video(video_path: str, max_frames: int | None = None) -> list[Image.Image]:
    """Extract frames from a video file as PIL Images."""
    frames = []
    try:
        # Use ffmpeg to extract frames
        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "frame_%06d.png")
            cmd = ["ffmpeg", "-i", video_path, "-vsync", "0"]
            if max_frames is not None:
                cmd.extend(["-vframes", str(max_frames)])
            cmd.extend([pattern])
            subprocess.run(cmd, check=True, capture_output=True)

            # Read extracted frames
            idx = 1
            while True:
                frame_path = pattern % idx
                if not os.path.exists(frame_path):
                    break
                frames.append(Image.open(frame_path).convert("RGB"))
                idx += 1
    except Exception as e:
        print(f"Warning: Failed to extract frames from video: {e}")
    return frames


def _get_video_frame_count(video_path: str) -> int:
    """Get the number of frames in a video file."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip())
    except Exception:
        # Fallback: extract all frames and count
        frames = _extract_frames_from_video(video_path)
        return len(frames)


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


def _process_pipeline_output(output, num_frames: int, height: int, width: int,
                             fps: float, seed: int, generation_time: float) -> dict[str, Any]:
    """Common output processing for all generation functions."""
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

    # Convert to numpy - keep as uint8 since decoder already outputs [0, 255]
    video_np = video_tensor.cpu().numpy()

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
    image_start: Image.Image | None = None,
    image_end: Image.Image | None = None,
    image_strength: float = 1.0,
    source_video_path: str | None = None,
    mask_tensor: torch.Tensor | None = None,
    mask_strength: float = 1.0,
    mask_start_frame: int = 0,
    lora_entries: list | None = None,
    fp8: bool = False,
) -> dict[str, Any]:
    """Generate video from text prompt. Supports masking/inpainting and LoRA."""
    global pipeline, models, pipeline_manager

    if pipeline_manager is None:
        init_pipeline(MODEL_VARIANT)

    # Get pipeline (handles LoRA switching)
    active_pipeline = pipeline_manager.get_pipeline(
        mode="text_to_video", fp8=fp8, lora_entries=lora_entries
    )

    device = torch.device("cuda")

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

    # Build image conditioning list (following Wan2GP pattern)
    # Latent stride is typically 8 for the video VAE temporal compression
    latent_stride = 8
    images = []
    images_stage2 = []

    # Clamp image_strength to valid range
    image_strength = max(0.0, min(1.0, float(image_strength)))

    if isinstance(active_pipeline, TI2VidTwoStagesPipeline):
        # For two-stage pipeline, we need both images and images_stage2
        if image_start is not None:
            entry = (image_start, _to_latent_index(0, latent_stride), image_strength, "lanczos")
            images.append(entry)
            images_stage2.append(entry)
            print(f"  Start frame conditioning: latent_idx=0, strength={image_strength}")

        if image_end is not None:
            entry = (image_end, _to_latent_index(num_frames - 1, latent_stride), 1.0)
            images.append(entry)
            images_stage2.append(entry)
            print(f"  End frame conditioning: latent_idx={_to_latent_index(num_frames - 1, latent_stride)}, strength=1.0")
    else:
        # For distilled pipeline, single images list
        if image_start is not None:
            images.append((image_start, _to_latent_index(0, latent_stride), image_strength, "lanczos"))
            print(f"  Start frame conditioning: latent_idx=0, strength={image_strength}")

        if image_end is not None:
            images.append((image_end, _to_latent_index(num_frames - 1, latent_stride), 1.0))
            print(f"  End frame conditioning: latent_idx={_to_latent_index(num_frames - 1, latent_stride)}, strength=1.0")

    # Build masking source for inpainting
    masking_source = None
    masking_strength_val = None
    if source_video_path and mask_tensor is not None:
        masking_source = {
            "video": source_video_path,
            "mask": mask_tensor,
            "start_frame": mask_start_frame,
        }
        masking_strength_val = float(mask_strength)
        print(f"  Masking: strength={mask_strength}, start_frame={mask_start_frame}")

    # Generate video
    print(f"Generating video: {prompt[:60]}...")
    print(f"  Resolution: {target_width}x{target_height}, Frames: {num_frames}, Seed: {seed}")
    start_time = time.time()

    if isinstance(active_pipeline, TI2VidTwoStagesPipeline):
        neg_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT
        output = active_pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=int(seed),
            height=target_height,
            width=target_width,
            num_frames=int(num_frames),
            frame_rate=float(fps),
            num_inference_steps=int(num_inference_steps),
            cfg_guidance_scale=float(guidance_scale),
            images=images,
            images_stage2=images_stage2 if images_stage2 else None,
            tiling_config=tiling_config,
            enhance_prompt=False,
            audio_conditionings=audio_conditionings,
            masking_source=masking_source,
            masking_strength=masking_strength_val,
        )
    else:
        output = active_pipeline(
            prompt=prompt,
            seed=int(seed),
            height=target_height,
            width=target_width,
            num_frames=int(num_frames),
            frame_rate=float(fps),
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=False,
            audio_conditionings=audio_conditionings,
            masking_source=masking_source,
            masking_strength=masking_strength_val,
        )

    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.1f}s")

    return _process_pipeline_output(output, num_frames, height, width, fps, seed, generation_time)


def generate_controlnet_video(
    prompt: str,
    control_type: str,
    control_video_path: str,
    control_strength: float = 1.0,
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    num_frames: int = 121,
    height: int = 512,
    width: int = 768,
    fps: float = DEFAULT_FPS,
    seed: int = 0,
    tile_size: int | None = None,
    image_start: Image.Image | None = None,
    image_end: Image.Image | None = None,
    image_strength: float = 1.0,
    audio_data: bytes | None = None,
    audio_strength: float = 1.0,
    lora_entries: list | None = None,
    fp8: bool = False,
) -> dict[str, Any]:
    """Generate video with IC-LoRA controlnet conditioning."""
    global pipeline_manager

    if pipeline_manager is None:
        init_pipeline(MODEL_VARIANT)

    active_pipeline = pipeline_manager.get_pipeline(
        mode="controlnet", fp8=fp8, control_type=control_type, lora_entries=lora_entries
    )

    # Align dimensions
    target_height = int(height)
    target_width = int(width)
    if target_height % 64 != 0:
        target_height = int(math.ceil(target_height / 64) * 64)
    if target_width % 64 != 0:
        target_width = int(math.ceil(target_width / 64) * 64)

    tiling_config = _build_tiling_config(tile_size, fps)

    # Audio conditioning
    audio_conditionings = None
    if audio_data is not None and audio_strength > 0.0:
        audio_latent = encode_audio_to_latent(audio_data, num_frames, fps)
        if audio_latent is not None:
            audio_conditionings = [AudioConditionByLatent(audio_latent, audio_strength)]

    # Build image conditionings
    latent_stride = 8
    images = []
    image_strength = max(0.0, min(1.0, float(image_strength)))
    if image_start is not None:
        images.append((image_start, _to_latent_index(0, latent_stride), image_strength, "lanczos"))
    if image_end is not None:
        images.append((image_end, _to_latent_index(num_frames - 1, latent_stride), 1.0))

    # Build video conditioning for IC-LoRA
    video_conditioning = [(control_video_path, float(control_strength))]

    print(f"Generating controlnet video ({control_type}): {prompt[:60]}...")
    print(f"  Resolution: {target_width}x{target_height}, Frames: {num_frames}, Seed: {seed}")
    start_time = time.time()

    output = active_pipeline(
        prompt=prompt,
        seed=int(seed),
        height=target_height,
        width=target_width,
        num_frames=int(num_frames),
        frame_rate=float(fps),
        images=images,
        video_conditioning=video_conditioning,
        tiling_config=tiling_config,
        enhance_prompt=False,
        audio_conditionings=audio_conditionings,
    )

    generation_time = time.time() - start_time
    print(f"ControlNet generation completed in {generation_time:.1f}s")

    return _process_pipeline_output(output, num_frames, height, width, fps, seed, generation_time)


def generate_keyframe_video(
    prompt: str,
    keyframes: list[tuple],
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    num_frames: int = 121,
    height: int = 512,
    width: int = 768,
    fps: float = DEFAULT_FPS,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int = 0,
    tile_size: int | None = None,
    audio_data: bytes | None = None,
    audio_strength: float = 1.0,
    lora_entries: list | None = None,
    fp8: bool = False,
) -> dict[str, Any]:
    """Generate video by interpolating between keyframes."""
    global pipeline_manager

    if pipeline_manager is None:
        init_pipeline(MODEL_VARIANT)

    active_pipeline = pipeline_manager.get_pipeline(
        mode="keyframe_interpolation", fp8=fp8, lora_entries=lora_entries
    )

    # Align dimensions
    target_height = int(height)
    target_width = int(width)
    if target_height % 64 != 0:
        target_height = int(math.ceil(target_height / 64) * 64)
    if target_width % 64 != 0:
        target_width = int(math.ceil(target_width / 64) * 64)

    tiling_config = _build_tiling_config(tile_size, fps)

    # Audio conditioning
    audio_conditionings = None
    if audio_data is not None and audio_strength > 0.0:
        audio_latent = encode_audio_to_latent(audio_data, num_frames, fps)
        if audio_latent is not None:
            audio_conditionings = [AudioConditionByLatent(audio_latent, audio_strength)]

    neg_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT

    print(f"Generating keyframe interpolation video: {prompt[:60]}...")
    print(f"  Resolution: {target_width}x{target_height}, Frames: {num_frames}, Seed: {seed}")
    print(f"  Keyframes: {len(keyframes)} frames")
    start_time = time.time()

    output = active_pipeline(
        prompt=prompt,
        negative_prompt=neg_prompt,
        seed=int(seed),
        height=target_height,
        width=target_width,
        num_frames=int(num_frames),
        frame_rate=float(fps),
        num_inference_steps=int(num_inference_steps),
        cfg_guidance_scale=float(guidance_scale),
        images=keyframes,
        tiling_config=tiling_config,
        enhance_prompt=False,
        audio_conditionings=audio_conditionings,
    )

    generation_time = time.time() - start_time
    print(f"Keyframe interpolation completed in {generation_time:.1f}s")

    return _process_pipeline_output(output, num_frames, height, width, fps, seed, generation_time)


def generate_video_extension(
    prompt: str,
    source_video_path: str,
    num_continuation_frames: int = 61,
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    height: int = 512,
    width: int = 768,
    fps: float = DEFAULT_FPS,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int = 0,
    tile_size: int | None = None,
    audio_data: bytes | None = None,
    audio_strength: float = 1.0,
    lora_entries: list | None = None,
    fp8: bool = False,
) -> dict[str, Any]:
    """Extend a video by generating continuation frames.

    Uses masking: source frames are preserved (mask=0), continuation frames
    are generated (mask=1). The last source frame is used as an image
    conditioning at the boundary for visual continuity.
    """
    global pipeline_manager

    if pipeline_manager is None:
        init_pipeline(MODEL_VARIANT)

    # Count source video frames
    source_frame_count = _get_video_frame_count(source_video_path)
    if source_frame_count <= 0:
        return {"error": "Could not read source video frames"}

    # Compute total frames aligned to 8k+1
    total_frames = _align_frames(source_frame_count + num_continuation_frames)
    actual_continuation = total_frames - source_frame_count

    # Build temporal mask: 0 for source frames (preserve), 1 for continuation (generate)
    # Shape (1, T, 1, 1) so _coerce_mask_tensor maps it to (1, 1, T, 1, 1) which
    # broadcasts spatially across all pixels per frame.
    mask = torch.zeros(1, total_frames, 1, 1, dtype=torch.float32)
    mask[0, source_frame_count:, 0, 0] = 1.0

    # Extract last source frame for boundary conditioning
    source_frames = _extract_frames_from_video(source_video_path)
    boundary_image = source_frames[-1] if source_frames else None

    print(f"Video extension: {source_frame_count} source + {actual_continuation} new = {total_frames} total frames")

    # Use generate_video with masking
    return generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=total_frames,
        height=height,
        width=width,
        fps=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        audio_data=audio_data,
        audio_strength=audio_strength,
        tile_size=tile_size,
        image_start=boundary_image,
        image_strength=1.0,
        source_video_path=source_video_path,
        mask_tensor=mask,
        mask_strength=1.0,
        mask_start_frame=0,
        lora_entries=lora_entries,
        fp8=fp8,
    )


def generate_audio_for_video(
    prompt: str,
    source_video_path: str,
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    fps: float = DEFAULT_FPS,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    seed: int = 0,
    tile_size: int | None = None,
    lora_entries: list | None = None,
    fp8: bool = False,
) -> dict[str, Any]:
    """Generate audio for an existing video (foley/V2A).

    Conditions all video frames as image conditionings with strength=1.0
    so the video stays fixed while audio is generated by the joint model.
    """
    global pipeline_manager

    if pipeline_manager is None:
        init_pipeline(MODEL_VARIANT)

    # Extract frames from source video
    source_frames = _extract_frames_from_video(source_video_path)
    if not source_frames:
        return {"error": "Could not extract frames from source video"}

    num_frames = len(source_frames)
    # Get dimensions from first frame
    width, height = source_frames[0].size

    active_pipeline = pipeline_manager.get_pipeline(
        mode="video_to_audio", fp8=fp8, lora_entries=lora_entries
    )

    # Align dimensions
    target_height = int(height)
    target_width = int(width)
    if target_height % 64 != 0:
        target_height = int(math.ceil(target_height / 64) * 64)
    if target_width % 64 != 0:
        target_width = int(math.ceil(target_width / 64) * 64)

    tiling_config = _build_tiling_config(tile_size, fps)

    # Build image conditionings from all source frames at their respective positions
    latent_stride = 8
    images = []
    for frame_idx in range(0, num_frames, latent_stride):
        if frame_idx < len(source_frames):
            latent_idx = _to_latent_index(frame_idx, latent_stride)
            images.append((source_frames[frame_idx], latent_idx, 1.0))

    # Also add first and last frames to ensure coverage
    if source_frames:
        images.insert(0, (source_frames[0], 0, 1.0, "lanczos"))
        last_latent_idx = _to_latent_index(num_frames - 1, latent_stride)
        images.append((source_frames[-1], last_latent_idx, 1.0))

    print(f"Generating audio for video: {prompt[:60]}...")
    print(f"  Source: {num_frames} frames, {len(images)} conditionings")
    start_time = time.time()

    if isinstance(active_pipeline, TI2VidTwoStagesPipeline):
        neg_prompt = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT
        output = active_pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            seed=int(seed),
            height=target_height,
            width=target_width,
            num_frames=int(num_frames),
            frame_rate=float(fps),
            num_inference_steps=int(num_inference_steps),
            cfg_guidance_scale=float(guidance_scale),
            images=images,
            images_stage2=images,
            tiling_config=tiling_config,
            enhance_prompt=False,
        )
    else:
        output = active_pipeline(
            prompt=prompt,
            seed=int(seed),
            height=target_height,
            width=target_width,
            num_frames=int(num_frames),
            frame_rate=float(fps),
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=False,
        )

    generation_time = time.time() - start_time
    print(f"Audio generation completed in {generation_time:.1f}s")

    return _process_pipeline_output(output, num_frames, height, width, fps, seed, generation_time)


def video_to_base64(video_np: np.ndarray, fps: float = 24.0, audio_np: np.ndarray | None = None, audio_sr: int = 24000) -> str:
    """Convert video numpy array to base64 encoded MP4."""
    # Normalize video to uint8
    if video_np.dtype != np.uint8:
        # Check actual range and normalize accordingly
        min_val = float(video_np.min())
        max_val = float(video_np.max())
        if max_val <= 1.0 and min_val >= -1.0:
            # Range is [-1, 1]
            video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        elif max_val <= 1.0 and min_val >= 0.0:
            # Range is [0, 1]
            video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
        else:
            # Already in [0, 255] range (float), just convert
            video_np = video_np.clip(0, 255).astype(np.uint8)

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

        # Image conditioning (start and end frames)
        image_start = _decode_base64_image(job_input.get("image_start"))
        image_end = _decode_base64_image(job_input.get("image_end"))
        image_strength = job_input.get("image_strength", 1.0)

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
            image_start=image_start,
            image_end=image_end,
            image_strength=image_strength,
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
            "has_image_start": image_start is not None,
            "has_image_end": image_end is not None,
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
