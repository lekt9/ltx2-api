"""Gemma text encoder base class (simplified without mmgp dependency)."""
import functools
import os
from pathlib import Path
from typing import NamedTuple

import torch
from accelerate import init_empty_weights
from einops import rearrange
from safetensors.torch import load_file
from transformers import AutoImageProcessor, Gemma3ForCausalLM, Gemma3Processor, AutoConfig

from ....loader.module_ops import ModuleOps
from ..embeddings_connector import Embeddings1DConnector
from ..feature_extractor import GemmaFeaturesExtractorProjLinear
from ..tokenizer import LTXVGemmaTokenizer
from shared.utils import files_locator as fl


# Gemma folder name
_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"


class GemmaTextEncoderModelBase(torch.nn.Module):
    """
    Gemma Text Encoder Model.
    This base class combines the tokenizer, Gemma model and feature extractor to provide a preprocessing
    for implementation classes for multimodal pipelines.
    """

    def __init__(
        self,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear | None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForCausalLM | None = None,
        img_processor: Gemma3Processor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self._gemma_root = None
        self.tokenizer = tokenizer
        self.model = model
        self.processor = img_processor
        if feature_extractor_linear is not None:
            self.feature_extractor_linear = feature_extractor_linear.to(dtype=dtype)
        else:
            self.feature_extractor_linear = None

    def encode_raw(self, text: str, padding_side: str = "left") -> "RawTextEmbeddings":
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return RawTextEmbeddings(outputs.hidden_states, attention_mask, padding_side)

    def _run_feature_extractor(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "right"
    ) -> torch.Tensor:
        if self.feature_extractor_linear is None:
            raise ValueError("feature_extractor_linear is not available for this text encoder")
        encoded_text_features = torch.stack(hidden_states, dim=-1)
        encoded_text_features_dtype = encoded_text_features.dtype

        sequence_lengths = attention_mask.sum(dim=-1)
        normed_concated_encoded_text_features = _norm_and_concat_padded_batch(
            encoded_text_features, sequence_lengths, padding_side=padding_side
        )

        return self.feature_extractor_linear(normed_concated_encoded_text_features.to(encoded_text_features_dtype))

    def _convert_to_additive_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (attention_mask - 1).to(dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(dtype).max

    def _preprocess_text(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.encode_raw(text, padding_side=padding_side)
        attention_mask = raw.attention_mask
        projected = self._run_feature_extractor(
            hidden_states=raw.hidden_states, attention_mask=attention_mask, padding_side=padding_side
        )
        return projected, attention_mask

    def _init_image_processor(self) -> None:
        img_processor = AutoImageProcessor.from_pretrained(self._gemma_root, local_files_only=True)
        if not self.tokenizer:
            raise ValueError("Tokenizer is not loaded, cannot load image processor")
        self.processor = Gemma3Processor(image_processor=img_processor, tokenizer=self.tokenizer.tokenizer)

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 42,
    ) -> str:
        if self.processor is None:
            self._init_image_processor()
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[self.model.device]):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]):]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]
        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 42,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")

    def forward(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("This method is not implemented for the base class")


class RawTextEmbeddings(NamedTuple):
    hidden_states: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor
    padding_side: str


def _apply_feature_extractor(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    padding_side: str,
    feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
) -> torch.Tensor:
    encoded_text_features = torch.stack(hidden_states, dim=-1)
    sequence_lengths = attention_mask.sum(dim=-1)
    normed = _norm_and_concat_padded_batch(
        encoded_text_features,
        sequence_lengths,
        padding_side=padding_side,
    )
    # Convert to the feature extractor's weight dtype to avoid dtype mismatch
    target_dtype = feature_extractor_linear.aggregate_embed.weight.dtype
    return feature_extractor_linear(normed.to(target_dtype))


def _apply_connectors(
    encoded_input: torch.Tensor,
    attention_mask: torch.Tensor,
    embeddings_connector: Embeddings1DConnector,
    audio_embeddings_connector: Embeddings1DConnector,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    connector_attention_mask = (attention_mask - 1).to(encoded_input.dtype).reshape(
        (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
    ) * torch.finfo(encoded_input.dtype).max
    encoded, encoded_connector_attention_mask = embeddings_connector(
        encoded_input,
        connector_attention_mask,
    )
    attention_mask_out = (encoded_connector_attention_mask < 0.000001).to(torch.int64)
    attention_mask_out = attention_mask_out.reshape([encoded.shape[0], encoded.shape[1], 1])
    encoded = encoded * attention_mask_out
    encoded_for_audio, _ = audio_embeddings_connector(encoded_input, connector_attention_mask)
    return encoded, encoded_for_audio, attention_mask_out.squeeze(-1)


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding."""
    b, t, d, l = encoded_text.shape
    device = encoded_text.device

    token_indices = torch.arange(t, device=device)[None, :]

    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6

    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, t, -1)

    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def _preprocess_gemma_state_dict(sd: dict) -> dict:
    """Preprocess Gemma state dict by removing unused keys and mapping prefixes."""
    new_sd = {}
    sd.pop("spiece_model", None)

    for key, value in sd.items():
        new_key = key
        # Map language model prefix
        if new_key.startswith("model.language_model."):
            new_key = "model." + new_key[len("model.language_model."):]
        elif new_key.startswith("language_model."):
            new_key = "model." + new_key[len("language_model."):]

        # Skip vision tower and projector
        if "vision_tower" in new_key or "multi_modal_projector" in new_key:
            continue

        new_sd[new_key] = value

    return new_sd


def build_gemma_text_encoder(
    gemma_root: str, default_dtype: torch.dtype = torch.bfloat16
) -> GemmaTextEncoderModelBase:
    """Build a Gemma text encoder from a checkpoint path."""
    gemma_path = gemma_root
    if not gemma_path or not os.path.isfile(gemma_path):
        raise FileNotFoundError(f"Gemma checkpoint not found: {gemma_root}")

    # The gemma_path is like: .../gemma-3-12b-it-qat-q4_0-unquantized/gemma-3-12b-it-qat-q4_0-unquantized.safetensors
    # The directory containing it has the tokenizer and config files
    gemma_dir = os.path.dirname(gemma_path)

    # Try to find config in same directory, or fall back to files_locator
    config_path = os.path.join(gemma_dir, "config_light.json")
    if not os.path.isfile(config_path):
        config_path = fl.locate_file(os.path.join(_GEMMA_FOLDER, "config_light.json"), error_if_none=False)
        if not config_path:
            # Try standard transformers config
            config_path = os.path.join(gemma_dir, "config.json")

    # Tokenizer path is the directory containing tokenizer.json
    tokenizer_path = gemma_dir
    if not os.path.isfile(os.path.join(tokenizer_path, "tokenizer.json")):
        tokenizer_path = fl.locate_folder(_GEMMA_FOLDER, error_if_none=False)
        if not tokenizer_path:
            tokenizer_path = gemma_dir

    # Load config
    import json
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Remove model_type if present to avoid conflict with for_model()
    config_dict.pop("model_type", None)

    # Add required attributes for transformers 4.57+
    # layer_types: Pattern - every Nth layer uses global attention, others use sliding window
    if "layer_types" not in config_dict:
        num_layers = config_dict.get("num_hidden_layers", 48)
        sliding_window_pattern = config_dict.get("sliding_window_pattern", 6)
        layer_types = []
        for i in range(num_layers):
            if sliding_window_pattern > 0 and (i + 1) % sliding_window_pattern == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("sliding_attention")
        config_dict["layer_types"] = layer_types

    # use_bidirectional_attention: For causal LM, this should be False
    if "use_bidirectional_attention" not in config_dict:
        config_dict["use_bidirectional_attention"] = False

    # Disable cache for encoding (not generation) - avoids hybrid cache layer mismatch
    config_dict["use_cache"] = False
    config_dict.pop("cache_implementation", None)

    # Create model with config - create on CPU first then move to CUDA
    config = AutoConfig.for_model("gemma3", **config_dict)

    # Load state dict first
    sd = load_file(gemma_path)
    sd = _preprocess_gemma_state_dict(sd)

    # Convert to target dtype
    sd = {k: v.to(dtype=default_dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in sd.items()}

    # Create model with low_cpu_mem_usage to avoid OOM
    model = Gemma3ForCausalLM(config)
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda")
    model.eval()
    model.requires_grad_(False)

    # Create text encoder
    text_encoder = GemmaTextEncoderModelBase(
        feature_extractor_linear=None,
        tokenizer=LTXVGemmaTokenizer(tokenizer_path, 1024),
        model=model,
        dtype=default_dtype,
    )
    text_encoder._gemma_root = gemma_dir

    return text_encoder


def module_ops_from_gemma_root(gemma_root: str) -> tuple[ModuleOps, ...]:
    """Create module ops for loading Gemma from a root path."""
    gemma_path = gemma_root
    gemma_dir = os.path.dirname(gemma_root)
    tokenizer_path = fl.locate_folder(_GEMMA_FOLDER)

    def load_gemma(module: GemmaTextEncoderModelBase) -> GemmaTextEncoderModelBase:
        config_path = fl.locate_file(os.path.join(_GEMMA_FOLDER, "config_light.json"))

        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Remove model_type if present to avoid conflict with for_model()
        config_dict.pop("model_type", None)

        # Add required attributes for transformers 4.57+
        if "layer_types" not in config_dict:
            num_layers = config_dict.get("num_hidden_layers", 48)
            sliding_window_pattern = config_dict.get("sliding_window_pattern", 6)
            layer_types = []
            for i in range(num_layers):
                if sliding_window_pattern > 0 and (i + 1) % sliding_window_pattern == 0:
                    layer_types.append("full_attention")
                else:
                    layer_types.append("sliding_attention")
            config_dict["layer_types"] = layer_types

        if "use_bidirectional_attention" not in config_dict:
            config_dict["use_bidirectional_attention"] = False

        # Disable cache for encoding (not generation) - avoids hybrid cache layer mismatch
        config_dict["use_cache"] = False
        config_dict.pop("cache_implementation", None)

        config = AutoConfig.for_model("gemma3", **config_dict)

        sd = load_file(gemma_path)
        sd = _preprocess_gemma_state_dict(sd)
        sd = {k: v.to(dtype=torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in sd.items()}

        model = Gemma3ForCausalLM(config)
        model.load_state_dict(sd, strict=False)
        model = model.to("cuda")
        model.eval()
        model.requires_grad_(False)

        module.model = model
        module._gemma_root = module._gemma_root or gemma_dir
        return module

    def load_tokenizer(module: GemmaTextEncoderModelBase) -> GemmaTextEncoderModelBase:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_path, 1024)
        module._gemma_root = module._gemma_root or gemma_dir
        return module

    gemma_load_ops = ModuleOps(
        "GemmaLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoderModelBase) and module.model is None,
        mutator=load_gemma,
    )
    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoderModelBase) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    return (gemma_load_ops, tokenizer_load_ops)


def encode_text(text_encoder: GemmaTextEncoderModelBase, prompts: list[str]) -> list[RawTextEmbeddings]:
    """Encode prompts with the Gemma text encoder, returning raw embeddings for later post-processing."""
    result = []
    for prompt in prompts:
        result.append(text_encoder.encode_raw(prompt))
    return result


def postprocess_text_embeddings(
    embeddings: list[RawTextEmbeddings],
    feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
    embeddings_connector: Embeddings1DConnector,
    audio_embeddings_connector: Embeddings1DConnector,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Apply projection and connector post-processing to raw embeddings."""
    projected = []
    for item in embeddings:
        attention_mask = item.attention_mask.to("cuda")
        encoded_input = _apply_feature_extractor(
            item.hidden_states,
            attention_mask,
            item.padding_side,
            feature_extractor_linear,
        )
        projected.append((encoded_input, attention_mask))
    results = []
    for encoded_input, attention_mask in projected:
        video_ctx, audio_ctx, _ = _apply_connectors(
            encoded_input,
            attention_mask,
            embeddings_connector,
            audio_embeddings_connector,
        )
        results.append((video_ctx, audio_ctx))
    return results


def resolve_text_connectors(
    text_encoder: GemmaTextEncoderModelBase,
    text_connectors: dict | None,
) -> tuple[GemmaFeaturesExtractorProjLinear, Embeddings1DConnector, Embeddings1DConnector]:
    if text_connectors is not None:
        feature_extractor = text_connectors.get("feature_extractor_linear")
        video_connector = text_connectors.get("embeddings_connector")
        audio_connector = text_connectors.get("audio_embeddings_connector")
    else:
        feature_extractor = getattr(text_encoder, "feature_extractor_linear", None)
        video_connector = getattr(text_encoder, "embeddings_connector", None)
        audio_connector = getattr(text_encoder, "audio_embeddings_connector", None)

    missing = []
    if feature_extractor is None:
        missing.append("feature_extractor_linear")
    if video_connector is None:
        missing.append("embeddings_connector")
    if audio_connector is None:
        missing.append("audio_embeddings_connector")
    if missing:
        raise ValueError(f"Missing text connector modules: {', '.join(missing)}")
    return feature_extractor, video_connector, audio_connector


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)
        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)

        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs
