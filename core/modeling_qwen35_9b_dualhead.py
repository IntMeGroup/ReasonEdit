# This module defines the release-time Qwen3.5 dual-head inference model.

from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig


class Qwen35DualHeadSFTModel(nn.Module):
    # This class wraps the multimodal generator and three-score regression head.
    def __init__(
        self,
        model_name_or_path: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: Optional[List[str]],
        precision: str,
        load_in_4bit: bool,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not target_modules:
            raise ValueError("target_modules must not be empty")
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in dtype_map:
            raise ValueError(f"Unsupported precision: {precision}")
        if device is not None and device.type == "cuda":
            if device.index is None:
                raise ValueError("CUDA device must have an explicit index")
            torch.cuda.set_device(device.index)

        load_kwargs = {
            "dtype": dtype_map[precision],
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if precision == "bf16" else dtype_map[precision],
                bnb_4bit_use_double_quant=True,
            )
            if device is not None:
                load_kwargs["device_map"] = {"": device.index if device.type == "cuda" else str(device)}
        base_model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **load_kwargs)
        if load_in_4bit:
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        self.model = get_peft_model(base_model, lora_config, autocast_adapter_dtype=False)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.model.config, "text_config"):
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Cannot infer hidden_size from model config")
        bottleneck = max(1, hidden_size // 2)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, 3),
        )
        self.regression_head = self.regression_head.to(dtype_map[precision])
        self._cached_norm_output = None

    # This method resolves the final text backbone norm used as the feature tap.
    def _get_feature_tap_module(self):
        backbone = self.model.get_base_model()
        return backbone.model.language_model.norm

    # This method caches the normalized hidden states during one forward pass.
    @contextmanager
    def _capture_feature_tap(self):
        self._cached_norm_output = None

        def hook(_module, _inputs, output):
            self._cached_norm_output = output

        handle = self._get_feature_tap_module().register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    # This method predicts regression scores from one hidden-state anchor position.
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        score_token_positions: Optional[torch.Tensor] = None,
    ):
        if score_token_positions is None:
            raise ValueError("score_token_positions is required")
        with self._capture_feature_tap():
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
                logits_to_keep=0,
            )
        hidden_states = self._cached_norm_output
        if hidden_states is None:
            raise ValueError("Failed to capture final normalized hidden states")
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        clamped_positions = torch.clamp(score_token_positions, min=0, max=hidden_states.size(1) - 1)
        pooled = hidden_states[batch_indices, clamped_positions]
        pooled = pooled.to(self.regression_head[0].weight.dtype)
        return self.regression_head(pooled)
