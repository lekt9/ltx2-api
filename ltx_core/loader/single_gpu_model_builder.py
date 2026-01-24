"""Simplified single GPU model builder for LTX-2 (without mmgp dependency)."""
import logging
from dataclasses import dataclass, field, replace
from typing import Generic

import torch
from safetensors.torch import load_file

from .module_ops import ModuleOps
from .primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from .registry import DummyRegistry, Registry
from .sd_ops import SDOps
from .sft_loader import SafetensorsModelStateDictLoader
from ..model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    Simplified version without mmgp dependency.
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry, repr=False)
    shared_state_dict: dict | None = field(default=None, repr=False)
    shared_quantization_map: dict | None = field(default=None, repr=False)
    shared_config: dict | None = field(default=None, repr=False)
    ignore_missing_keys: bool = False
    copy_shared_state_dict: bool = False
    consume_shared_state_dict: bool = False

    def lora(self, lora_path: str, strength: float = 1.0, sd_ops: SDOps | None = None) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def model_config(self) -> dict:
        if self.shared_config is not None:
            return self.shared_config
        first_shard_path = self.model_path[0] if isinstance(self.model_path, tuple) else self.model_path
        return self.model_loader.metadata(first_shard_path)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        with torch.device("meta"):
            model = self.model_class_configurator.from_config(config)
        for module_op in module_ops:
            if module_op.matcher(model):
                model = module_op.mutator(model)
        return model

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        state_dict = registry.get(paths, sd_ops)
        if state_dict is None:
            state_dict = self.model_loader.load(paths, sd_ops=sd_ops, device=device)
            registry.add(paths, sd_ops=sd_ops, state_dict=state_dict)
        return state_dict

    def _filter_state_dict(self, state_dict: dict, sd_ops: SDOps | None) -> dict:
        if sd_ops is None:
            return dict(state_dict)
        filtered = {}
        if self.consume_shared_state_dict:
            for key in list(state_dict.keys()):
                expected_name = sd_ops.apply_to_key(key)
                if expected_name is None:
                    continue
                value = state_dict.pop(key)
                key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                for new_key, new_value in key_value_pairs:
                    filtered[new_key] = new_value
        else:
            for key, value in state_dict.items():
                expected_name = sd_ops.apply_to_key(key)
                if expected_name is None:
                    continue
                key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                for new_key, new_value in key_value_pairs:
                    filtered[new_key] = new_value
        return filtered

    def _return_model(self, meta_model: ModelType, device: torch.device) -> ModelType:
        uninitialized_params = [name for name, param in meta_model.named_parameters() if str(param.device) == "meta"]
        uninitialized_buffers = [name for name, buffer in meta_model.named_buffers() if str(buffer.device) == "meta"]
        if uninitialized_params or uninitialized_buffers:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized_params + uninitialized_buffers}")
            return meta_model
        retval = meta_model.to(device)
        return retval

    def _preprocess_sd(self, sd: dict) -> dict:
        """Preprocess state dict by removing common prefixes."""
        new_sd = {}
        prefixes = ["vae", "audio_vae", "vocoder", "text_embedding_projection", "diffusion_model", "model"]
        for k, v in sd.items():
            # Remove model. prefix
            if k.startswith("model."):
                k = k[len("model."):]
            # Try removing other prefixes
            for prefix in prefixes:
                if k.startswith(prefix + "."):
                    k = k[len(prefix) + 1:]
                    break
            new_sd[k] = v
        return new_sd

    def build(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> ModelType:
        device = torch.device("cuda") if device is None else device
        dtype = dtype or torch.bfloat16
        config = self.model_config()
        meta_model = self.meta_model(config, self.module_ops)

        if self.shared_state_dict is not None:
            sd = self._filter_state_dict(self.shared_state_dict, self.model_sd_ops)
            if self.copy_shared_state_dict:
                sd = {key: value.clone() if torch.is_tensor(value) else value for key, value in sd.items()}
            if len(sd):
                # Convert to target dtype and load
                sd = {k: v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in sd.items()}
                meta_model.load_state_dict(sd, strict=not self.ignore_missing_keys, assign=True)
            return self._return_model(meta_model, device)

        # Load from file(s)
        model_paths = self.model_path if isinstance(self.model_path, tuple) else [self.model_path]

        # Load and merge state dicts from all paths
        merged_sd = {}
        for path in model_paths:
            sd = load_file(path)
            merged_sd.update(sd)

        # Preprocess the state dict
        sd = self._preprocess_sd(merged_sd)

        # Apply sd_ops if provided
        if self.model_sd_ops is not None:
            sd = self._filter_state_dict(sd, self.model_sd_ops)

        # Convert to target dtype
        sd = {k: v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v for k, v in sd.items()}

        # Load into model
        meta_model.load_state_dict(sd, strict=not self.ignore_missing_keys, assign=True)

        return self._return_model(meta_model, device)
