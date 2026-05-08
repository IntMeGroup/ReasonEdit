# This module provides reusable inference for the CLI entrypoint.

import csv
import json
import multiprocessing as mp
import os
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from threading import Thread
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from peft import set_peft_model_state_dict
from PIL import Image, UnidentifiedImageError
from safetensors.torch import load_file
from scipy.stats import kendalltau
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from .calibration import AffineScoreCalibrator, SCORE_DIMENSIONS
from .modeling_qwen35_9b_dualhead import Qwen35DualHeadSFTModel
from .prompting import DEFAULT_SCORE_ANCHOR, RECOMMENDED_BASE_MODEL_PATH, build_prompt, resolve_existing_path, validate_batch_csv


BATCH_OUTPUT_FIELDNAMES = [
    "source",
    "edited",
    "instruction",
    "v_score",
    "e_score",
    "p_score",
    "cot_text",
]

DEFAULT_PIXEL_BUDGET = 262144
CSV_FLUSH_INTERVAL = 200
DEFAULT_LORA_RANK = 64
DEFAULT_LORA_ALPHA = 128
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = [
    "up_proj",
    "gate_proj",
    "attn.qkv",
    "patch_embed.proj",
    "k_proj",
    "q_proj",
    "merger.linear_fc2",
    "mlp.linear_fc1",
    "down_proj",
    "o_proj",
    "mlp.linear_fc2",
    "attn.proj",
    "merger.linear_fc1",
    "v_proj",
]
DEFAULT_PRECISION = "bf16"
DEFAULT_LOAD_IN_4BIT = True
SCORE_LABEL_ALIASES = (
    ("v_score", "e_score", "p_score"),
    ("visual_quality", "editing_alignment", "content_preservation"),
    ("logicality_score", "accuracy_score", "usefulness_score"),
)


def display_path_text(path_text: str, workspace_root: str) -> str:
    # This function formats a path for output without leaking absolute machine paths.
    raw_path = Path(str(path_text).strip())
    if raw_path.is_absolute():
        return os.path.relpath(raw_path.resolve(), Path(workspace_root).resolve())
    return str(raw_path)


def resolve_output_path(path_text: str, workspace_root: str) -> Path:
    # This function resolves one writable output path relative to the workspace root.
    raw_path = Path(str(path_text).strip())
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (Path(workspace_root) / raw_path).resolve()


def build_batch_output_row(item: Dict) -> Dict[str, object]:
    # This function converts one inference result payload into one CSV output row.
    return {
        "source": item["source"],
        "edited": item["edited"],
        "instruction": item["instruction"],
        "v_score": item.get("scores", {}).get("visual_quality", ""),
        "e_score": item.get("scores", {}).get("editing_alignment", ""),
        "p_score": item.get("scores", {}).get("content_preservation", ""),
        "cot_text": item.get("cot_text") or "",
    }


def select_label_columns(dataframe: pd.DataFrame) -> Optional[List[str]]:
    # This function finds a compatible three-column label schema in a batch CSV.
    for candidate in SCORE_LABEL_ALIASES:
        if set(candidate).issubset(dataframe.columns):
            return list(candidate)
    return None


def safe_corr_np(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    # This function computes PLCC, SRCC, and KRCC for one score dimension.
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return {"plcc": 0.0, "srcc": 0.0, "krcc": 0.0}
    plcc = float(np.corrcoef(x, y)[0, 1])
    ranked_x = pd.Series(x).rank(method="average").to_numpy()
    ranked_y = pd.Series(y).rank(method="average").to_numpy()
    srcc = float(np.corrcoef(ranked_x, ranked_y)[0, 1])
    krcc = float(kendalltau(x, y, nan_policy="omit").correlation)
    if np.isnan(plcc):
        plcc = 0.0
    if np.isnan(srcc):
        srcc = 0.0
    if np.isnan(krcc):
        krcc = 0.0
    return {"plcc": plcc, "srcc": srcc, "krcc": krcc}


def summarize_triplet_predictions(predictions: np.ndarray, labels: np.ndarray, dimension_names: List[str]) -> Dict[str, float]:
    # This function summarizes one three-dimensional prediction task.
    if predictions.shape != labels.shape:
        raise ValueError(f"Mismatched prediction and label shapes: {predictions.shape} vs {labels.shape}")
    if predictions.shape[1] != len(dimension_names):
        raise ValueError("dimension_names must match the number of score columns")
    metrics = {"sample_count": int(predictions.shape[0])}
    plcc_values = []
    srcc_values = []
    krcc_values = []
    for index, dimension_name in enumerate(dimension_names):
        corr = safe_corr_np(predictions[:, index], labels[:, index])
        metrics[f"{dimension_name}_plcc"] = float(corr["plcc"])
        metrics[f"{dimension_name}_srcc"] = float(corr["srcc"])
        metrics[f"{dimension_name}_krcc"] = float(corr["krcc"])
        plcc_values.append(corr["plcc"])
        srcc_values.append(corr["srcc"])
        krcc_values.append(corr["krcc"])
    metrics["mean_plcc"] = float(np.mean(plcc_values))
    metrics["mean_srcc"] = float(np.mean(srcc_values))
    metrics["mean_krcc"] = float(np.mean(krcc_values))
    return metrics


def is_severe_inference_error(error: Exception) -> bool:
    # This function identifies runtime failures that should stop the whole batch.
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    severe_terms = [
        "out of memory",
        "cuda error",
        "cuda failure",
        "nccl",
        "cublas",
        "cudnn",
        "device-side assert",
    ]
    return any(term in text for term in severe_terms)


def build_failed_batch_result(
    source_path: str,
    edited_path: str,
    instruction: str,
    workspace_root: Optional[str] = None,
) -> Dict:
    # This function builds one CSV-compatible failure result for recoverable row-level errors.
    return {
        "source": display_path_text(source_path, workspace_root or "."),
        "edited": display_path_text(edited_path, workspace_root or "."),
        "instruction": str(instruction).strip(),
        "scores": {},
        "raw_scores": {},
        "cot_text": "",
    }


class CancelledInferenceError(RuntimeError):
    # This exception marks one inference job as intentionally cancelled by the user.
    pass


class EventStoppingCriteria(StoppingCriteria):
    # This stopping criterion halts generation when the caller sets one stop event.
    def __init__(self, stop_event: Optional[Event]):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return bool(self.stop_event is not None and self.stop_event.is_set())


@dataclass
class ReleaseModelSpec:
    # This dataclass stores one release model package definition.
    model_id: str
    display_name: str
    description: str
    recommended_base_model_path: str
    lora_path: str


class DualHeadInferenceEngine:
    # This class loads one release package and runs score-only or score-plus-CoT inference.
    def __init__(
        self,
        workspace_root: str,
        registry_path: str,
        lora_id: Optional[str] = None,
        lora_path: Optional[str] = None,
        base_model_path: Optional[str] = None,
        device: Optional[str] = None,
        load_on_init: bool = True,
    ):
        self.workspace_root_display = str(workspace_root).strip() or "ReasonEdit"
        self.workspace_root = str(Path(self.workspace_root_display).resolve())
        self.registry_path = Path(registry_path).resolve()
        self.registry = self._load_registry()
        self.model_spec = self._resolve_model_spec(lora_id=lora_id, lora_path=lora_path)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.base_model_path_display = str(
            base_model_path or self.model_spec.recommended_base_model_path or RECOMMENDED_BASE_MODEL_PATH
        ).strip()
        self.base_model_path = str(resolve_existing_path(self.base_model_path_display, self.workspace_root))
        self.lora_path_display = str(self.model_spec.lora_path).strip()
        self.checkpoint_dir = resolve_existing_path(self.lora_path_display, self.workspace_root)
        self.calibrator = AffineScoreCalibrator(self.checkpoint_dir)
        self.runtime_config = {
            "lora_rank": DEFAULT_LORA_RANK,
            "lora_alpha": DEFAULT_LORA_ALPHA,
            "lora_dropout": DEFAULT_LORA_DROPOUT,
            "lora_target_modules": DEFAULT_LORA_TARGET_MODULES,
            "precision": DEFAULT_PRECISION,
            "load_in_4bit": DEFAULT_LOAD_IN_4BIT,
        }
        self.processor = None
        self.model = None
        if load_on_init:
            self._ensure_loaded()

    # This method emits one optional progress event to the caller.
    def _report_progress(self, progress_callback, progress: float, stage: str, message: str):
        if progress_callback is not None:
            progress_callback(progress, stage, message)

    # This method loads the release model registry.
    def _load_registry(self) -> Dict:
        with self.registry_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    # This method returns one resolved model package from registry or direct path input.
    def _resolve_model_spec(self, lora_id: Optional[str], lora_path: Optional[str]) -> ReleaseModelSpec:
        if lora_path:
            resolved_lora_path = Path(lora_path)
            return ReleaseModelSpec(
                model_id="custom",
                display_name=resolved_lora_path.name,
                description="User supplied LoRA package path.",
                recommended_base_model_path=self.registry.get("recommended_base_model_path", RECOMMENDED_BASE_MODEL_PATH),
                lora_path=str(lora_path),
            )
        available = self.registry.get("models", {})
        selected_id = lora_id or self.registry.get("default_model_id")
        if selected_id not in available:
            raise ValueError(f"Unknown lora_id: {selected_id}")
        item = available[selected_id]
        package_name = Path(item["lora_path"]).name
        return ReleaseModelSpec(
            model_id=selected_id,
            display_name=package_name,
            description=item.get("description", package_name),
            recommended_base_model_path=item.get(
                "recommended_base_model_path",
                self.registry.get("recommended_base_model_path", RECOMMENDED_BASE_MODEL_PATH),
            ),
            lora_path=item["lora_path"],
        )

    # This method restores the release model weights and regression head.
    def _load_model(self, progress_callback=None) -> Qwen35DualHeadSFTModel:
        self._report_progress(progress_callback, 0.22, "loading", "Initializing model wrapper")
        model = Qwen35DualHeadSFTModel(
            model_name_or_path=self.base_model_path,
            lora_rank=int(self.runtime_config["lora_rank"]),
            lora_alpha=int(self.runtime_config["lora_alpha"]),
            lora_dropout=float(self.runtime_config["lora_dropout"]),
            target_modules=list(self.runtime_config["lora_target_modules"]),
            precision=self.runtime_config["precision"],
            load_in_4bit=bool(self.runtime_config["load_in_4bit"]),
            device=self.device,
        )
        self._report_progress(progress_callback, 0.80, "loading", "Loading LoRA adapter")
        adapter_path = self.checkpoint_dir / "adapter_model.safetensors"
        regression_head_path = self.checkpoint_dir / "regression_head.pt"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Missing adapter weights: {adapter_path}")
        if not regression_head_path.exists():
            raise FileNotFoundError(f"Missing regression head: {regression_head_path}")
        adapter_state = load_file(str(adapter_path), device="cpu")
        incompatible = set_peft_model_state_dict(model.model, adapter_state)
        unexpected_adapter_keys = list(incompatible.unexpected_keys)
        missing_adapter_keys = [
            key
            for key in incompatible.missing_keys
            if ("lora_" in key) or ("ranknum" in key) or ("lora_magnitude_vector" in key) or (".default" in key)
        ]
        if unexpected_adapter_keys or missing_adapter_keys:
            raise ValueError(f"Adapter key mismatch: {incompatible}")
        self._report_progress(progress_callback, 0.92, "loading", "Loading regression head")
        regression_state = torch.load(regression_head_path, map_location="cpu")
        model.regression_head.load_state_dict(regression_state)
        model.to(self.device)
        model.eval()
        self._report_progress(progress_callback, 1.00, "loading", "Model is ready")
        return model

    # This method lazily initializes processor and model weights.
    def _ensure_loaded(self, progress_callback=None):
        if self.processor is None:
            self._report_progress(progress_callback, 0.05, "loading", "Loading processor")
            self.processor = AutoProcessor.from_pretrained(str(self.checkpoint_dir), trust_remote_code=True)
            self._report_progress(progress_callback, 0.15, "loading", "Processor is ready")
        if self.model is None:
            self.model = self._load_model(progress_callback=progress_callback)

    # This method resolves the effective pixel budget for one request.
    def _resolve_pixel_budget(self, max_pixels_per_image: Optional[int]) -> int:
        if max_pixels_per_image is None or int(max_pixels_per_image) <= 0:
            return DEFAULT_PIXEL_BUDGET
        return int(max_pixels_per_image)

    # This method loads and resizes one RGB image under the saved pixel budget.
    def _load_rgb_image(self, image_path: str, max_pixels_per_image: Optional[int]) -> Image.Image:
        resolved = resolve_existing_path(image_path, self.workspace_root)
        max_pixels = self._resolve_pixel_budget(max_pixels_per_image)
        with Image.open(resolved) as image:
            image = image.convert("RGB")
            width, height = image.size
            pixels = width * height
            if max_pixels > 0 and pixels > max_pixels:
                scale = (max_pixels / float(pixels)) ** 0.5
                new_width = max(16, int(round(width * scale)))
                new_height = max(16, int(round(height * scale)))
                new_width = max(16, (new_width // 16) * 16)
                new_height = max(16, (new_height // 16) * 16)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return image

    # This method builds one prompt input for the processor.
    def _encode_prompt(
        self,
        source_path: str,
        edited_path: str,
        instruction: str,
        max_pixels_per_image: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        effective_pixels = self._resolve_pixel_budget(max_pixels_per_image)
        source_image = self._load_rgb_image(source_path, effective_pixels)
        edited_image = self._load_rgb_image(edited_path, effective_pixels)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": source_image},
                    {"type": "image", "image": edited_image},
                    {"type": "text", "text": build_prompt(instruction)},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "images_kwargs": {
                    "max_pixels": effective_pixels,
                    "min_pixels": effective_pixels,
                }
            },
        )

    # This method formats the displayed CoT tail using regression scores instead of generated scores.
    def _format_cot_display(self, cot_text: str, scores: List[float]) -> str:
        score_lines = [
            f"- Visual Quality: {float(scores[0]) * 100.0:.2f}",
            f"- Instruction Alignment: {float(scores[1]) * 100.0:.2f}",
            f"- Content Preservation: {float(scores[2]) * 100.0:.2f}",
        ]
        normalized = str(cot_text or "").strip()
        if DEFAULT_SCORE_ANCHOR in normalized:
            prefix = normalized.split(DEFAULT_SCORE_ANCHOR, 1)[0].rstrip()
            if prefix:
                return f"{prefix}\n\n{DEFAULT_SCORE_ANCHOR}\n" + "\n".join(score_lines)
            return f"{DEFAULT_SCORE_ANCHOR}\n" + "\n".join(score_lines)
        if normalized:
            return f"{normalized}\n\n{DEFAULT_SCORE_ANCHOR}\n" + "\n".join(score_lines)
        return f"{DEFAULT_SCORE_ANCHOR}\n" + "\n".join(score_lines)

    # This method moves processor tensors to the inference device.
    def _move_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    # This method returns the prefill output position immediately before the first generated token.
    def _get_anchor_position(self, prompt_inputs: Dict[str, torch.Tensor]) -> int:
        return prompt_inputs["input_ids"].size(1) - 1

    # This method converts one raw score triplet into raw and calibrated payloads.
    def _build_score_payloads(self, raw_scores: List[float]) -> Dict[str, Dict[str, float]]:
        if len(raw_scores) != len(SCORE_DIMENSIONS):
            raise ValueError(f"Expected {len(SCORE_DIMENSIONS)} regression scores, got {len(raw_scores)}")
        return {
            "raw_scores": self.calibrator.build_raw_score_dict(raw_scores),
            "scores": self.calibrator.calibrate_raw_scores(raw_scores),
        }

    # This method raises one explicit cancellation error when requested.
    def _raise_if_cancelled(self, stop_event: Optional[Event]):
        if stop_event is not None and stop_event.is_set():
            raise CancelledInferenceError("Inference cancelled by user.")

    # This method runs only the regression branch for maximum throughput.
    def predict_scores_only(
        self,
        source_path: str,
        edited_path: str,
        instruction: str,
        progress_callback=None,
        max_pixels_per_image: Optional[int] = None,
        stop_event: Optional[Event] = None,
    ) -> Dict:
        self._ensure_loaded(progress_callback=progress_callback)
        self._raise_if_cancelled(stop_event)
        self._report_progress(progress_callback, 0.62, "encoding", "Preparing images and prompt")
        prompt_inputs = self._move_inputs(
            self._encode_prompt(source_path, edited_path, instruction, max_pixels_per_image=max_pixels_per_image)
        )
        self._raise_if_cancelled(stop_event)
        score_token_position = self._get_anchor_position(prompt_inputs)
        self._report_progress(progress_callback, 0.82, "scoring", "Running regression head")
        with torch.inference_mode():
            raw_scores = self.model(
                input_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"],
                pixel_values=prompt_inputs["pixel_values"],
                image_grid_thw=prompt_inputs["image_grid_thw"],
                mm_token_type_ids=prompt_inputs["mm_token_type_ids"],
                score_token_positions=torch.tensor([score_token_position], dtype=torch.long, device=self.device),
            )[0].detach().float().cpu().tolist()
        self._raise_if_cancelled(stop_event)
        self._report_progress(progress_callback, 1.00, "done", "Inference completed")
        return self._format_result(source_path, edited_path, instruction, raw_scores, cot_text=None, mode="score_only")

    # This method runs generation and then reuses the same prompt-end anchor for regression.
    def predict_with_cot(
        self,
        source_path: str,
        edited_path: str,
        instruction: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        progress_callback=None,
        token_callback=None,
        max_pixels_per_image: Optional[int] = None,
        stop_event: Optional[Event] = None,
    ) -> Dict:
        self._ensure_loaded(progress_callback=progress_callback)
        self._raise_if_cancelled(stop_event)
        self._report_progress(progress_callback, 0.62, "encoding", "Preparing images and prompt")
        prompt_inputs = self._move_inputs(
            self._encode_prompt(source_path, edited_path, instruction, max_pixels_per_image=max_pixels_per_image)
        )
        self._raise_if_cancelled(stop_event)
        score_token_position = self._get_anchor_position(prompt_inputs)
        self._report_progress(progress_callback, 0.74, "scoring", "Running regression head")
        with torch.inference_mode():
            raw_scores = self.model(
                input_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"],
                pixel_values=prompt_inputs["pixel_values"],
                image_grid_thw=prompt_inputs["image_grid_thw"],
                mm_token_type_ids=prompt_inputs["mm_token_type_ids"],
                score_token_positions=torch.tensor([score_token_position], dtype=torch.long, device=self.device),
            )[0].detach().float().cpu().tolist()
        self._raise_if_cancelled(stop_event)
        calibrated_scores = self.calibrator.calibrate_triplet(raw_scores)
        self._report_progress(progress_callback, 0.80, "generation", "Generating CoT")
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            "input_ids": prompt_inputs["input_ids"],
            "attention_mask": prompt_inputs["attention_mask"],
            "pixel_values": prompt_inputs["pixel_values"],
            "image_grid_thw": prompt_inputs["image_grid_thw"],
            "mm_token_type_ids": prompt_inputs["mm_token_type_ids"],
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "temperature": temperature if temperature > 0.0 else None,
            "top_p": top_p,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([EventStoppingCriteria(stop_event)]),
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        error_holder = {}

        def generation_worker():
            try:
                with torch.inference_mode():
                    self.model.model.generate(**generation_kwargs)
            except Exception as error:
                error_holder["error"] = error

        generation_thread = Thread(target=generation_worker, daemon=True)
        generation_thread.start()
        cot_chunks: List[str] = []
        chunk_count = 0
        for chunk in streamer:
            self._raise_if_cancelled(stop_event)
            cot_chunks.append(chunk)
            chunk_count += 1
            if token_callback is not None:
                token_callback(chunk, "".join(cot_chunks))
            generation_progress = min(0.98, 0.80 + 0.006 * chunk_count)
            self._report_progress(progress_callback, generation_progress, "generation", "Generating CoT")
        generation_thread.join()
        if "error" in error_holder:
            raise error_holder["error"]
        self._raise_if_cancelled(stop_event)
        cot_text = self._format_cot_display("".join(cot_chunks).strip(), calibrated_scores)
        self._report_progress(progress_callback, 1.00, "done", "Inference completed")
        return self._format_result(source_path, edited_path, instruction, raw_scores, cot_text=cot_text, mode="cot")

    # This method runs one sample in the requested inference mode.
    def predict(
        self,
        source_path: str,
        edited_path: str,
        instruction: str,
        mode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        progress_callback=None,
        token_callback=None,
        max_pixels_per_image: Optional[int] = None,
        stop_event: Optional[Event] = None,
    ) -> Dict:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == "score_only":
            return self.predict_scores_only(
                source_path=source_path,
                edited_path=edited_path,
                instruction=instruction,
                progress_callback=progress_callback,
                max_pixels_per_image=max_pixels_per_image,
                stop_event=stop_event,
            )
        if normalized_mode in {"cot", "with_cot"}:
            return self.predict_with_cot(
                source_path=source_path,
                edited_path=edited_path,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                progress_callback=progress_callback,
                token_callback=token_callback,
                max_pixels_per_image=max_pixels_per_image,
                stop_event=stop_event,
            )
        raise ValueError(f"Unsupported mode: {mode}")

    # This method runs batch inference over one CSV file.
    def predict_csv(
        self,
        input_csv: str,
        output_csv: str,
        mode: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        row_callback=None,
        max_pixels_per_image: Optional[int] = None,
        stop_event: Optional[Event] = None,
    ) -> Dict:
        input_csv_display = str(input_csv)
        output_csv_display = str(output_csv)
        dataframe = pd.read_csv(resolve_existing_path(input_csv, self.workspace_root), encoding="utf-8")
        validate_batch_csv(dataframe)
        results: List[Dict] = []
        output_path = resolve_output_path(output_csv, self.workspace_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(dataframe)

        def flush_results():
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=BATCH_OUTPUT_FIELDNAMES)
                writer.writeheader()
                for item in results:
                    writer.writerow(build_batch_output_row(item))

        for row_index, (_, row) in enumerate(dataframe.iterrows(), start=1):
            self._raise_if_cancelled(stop_event)
            source_path = str(row["source"])
            edited_path = str(row["edited"])
            instruction = str(row["instruction"])
            try:
                result = self.predict(
                    source_path=source_path,
                    edited_path=edited_path,
                    instruction=instruction,
                    mode=mode,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_pixels_per_image=max_pixels_per_image,
                    stop_event=stop_event,
                )
            except CancelledInferenceError:
                raise
            except Exception as error:
                if is_severe_inference_error(error):
                    raise
                result = build_failed_batch_result(source_path, edited_path, instruction, workspace_root=self.workspace_root)
            results.append(result)
            if row_callback is not None:
                row_callback(row_index, total, result)
            if row_index % CSV_FLUSH_INTERVAL == 0:
                flush_results()
        flush_results()
        return {"num_samples": len(results), "input_csv": input_csv_display, "output_csv": output_csv_display, "results": results}

    # This method formats one standardized inference result payload.
    def _format_result(
        self,
        source_path: str,
        edited_path: str,
        instruction: str,
        raw_scores: List[float],
        cot_text: Optional[str],
        mode: str,
    ) -> Dict:
        score_payloads = self._build_score_payloads(raw_scores)
        return {
            "inference_status": "success",
            "error_message": "",
            "model_id": self.model_spec.model_id,
            "model_name": self.model_spec.display_name,
            "base_model_path": self.base_model_path_display,
            "lora_path": self.lora_path_display,
            "score_anchor": DEFAULT_SCORE_ANCHOR,
            "source": display_path_text(source_path, self.workspace_root),
            "edited": display_path_text(edited_path, self.workspace_root),
            "instruction": str(instruction).strip(),
            "mode": mode,
            "scores": score_payloads["scores"],
            "raw_scores": score_payloads["raw_scores"],
            "score_dimensions": list(SCORE_DIMENSIONS),
            "calibration": self.calibrator.describe(),
            "cot_text": cot_text,
        }

    # This method exposes compact configuration metadata for the CLI.
    def describe(self) -> Dict:
        return {
            "workspace_root": self.workspace_root_display,
            "base_model_path": self.base_model_path_display,
            "recommended_base_model_path": self.model_spec.recommended_base_model_path,
            "lora_path": self.lora_path_display,
            "default_model_id": self.registry.get("default_model_id"),
            "active_model_id": self.model_spec.model_id,
            "active_model_name": self.model_spec.display_name,
            "active_model_description": self.model_spec.description,
            "available_models": {
                model_id: {
                    **item,
                    "display_name": Path(item["lora_path"]).name,
                }
                for model_id, item in self.registry.get("models", {}).items()
            },
            "calibration": self.calibrator.describe(),
            "pixel_budget": DEFAULT_PIXEL_BUDGET,
            "mode_default": "cot",
        }


def batch_worker_loop(worker_id: int, device: str, engine_init_kwargs: Dict, request_payload: Dict, task_queue, event_queue):
    # This function runs one dedicated CLI batch inference worker on a single GPU.
    try:
        if str(device).startswith("cuda:"):
            torch.cuda.set_device(int(str(device).split(":")[1]))
        engine = DualHeadInferenceEngine(
            workspace_root=engine_init_kwargs["workspace_root"],
            registry_path=engine_init_kwargs["registry_path"],
            lora_id=engine_init_kwargs["lora_id"],
            lora_path=engine_init_kwargs["lora_path"],
            base_model_path=engine_init_kwargs["base_model_path"],
            device=device,
            load_on_init=False,
        )
        event_queue.put({"type": "worker_ready", "worker_id": worker_id, "device": device})
        while True:
            task = task_queue.get()
            if task is None:
                break
            event_queue.put(
                {
                    "type": "started",
                    "worker_id": worker_id,
                    "device": device,
                    "row_index": task["row_index"],
                    "source": task["source"],
                    "edited": task["edited"],
                }
            )
            try:
                result = engine.predict(
                    source_path=task["source"],
                    edited_path=task["edited"],
                    instruction=task["instruction"],
                    mode=request_payload["mode"],
                    max_new_tokens=int(request_payload["max_new_tokens"]),
                    temperature=float(request_payload["temperature"]),
                    top_p=float(request_payload["top_p"]),
                    max_pixels_per_image=int(request_payload["pixel_budget"]),
                )
            except Exception as error:
                if is_severe_inference_error(error):
                    raise
                result = build_failed_batch_result(
                    task["source"],
                    task["edited"],
                    task["instruction"],
                    workspace_root=engine.workspace_root,
                )
            event_queue.put(
                {
                    "type": "done",
                    "worker_id": worker_id,
                    "device": device,
                    "row_index": task["row_index"],
                    "result": result,
                }
            )
    except Exception as error:
        event_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "device": device,
                "row_index": -1,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )


def write_batch_results(output_path: Path, results: List[Dict]):
    # This function writes merged batch inference rows to CSV.
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=BATCH_OUTPUT_FIELDNAMES)
        csv_writer.writeheader()
        for item in results:
            csv_writer.writerow(build_batch_output_row(item))


def summarize_batch_csv(input_csv: Path, output_csv: Path) -> Dict[str, float]:
    # This function computes label-aware metrics for a completed batch output CSV.
    input_df = pd.read_csv(input_csv, encoding="utf-8")
    label_columns = select_label_columns(input_df)
    if label_columns is None:
        return {"sample_count": int(len(input_df))}
    output_df = pd.read_csv(output_csv, encoding="utf-8")
    prediction_columns = [column for column in ["v_score", "e_score", "p_score"] if column in output_df.columns]
    if len(prediction_columns) != 3:
        return {"sample_count": int(len(output_df))}
    aligned_labels = input_df[label_columns].astype(float).to_numpy()
    aligned_predictions = output_df[prediction_columns].astype(float).to_numpy()
    return summarize_triplet_predictions(aligned_predictions, aligned_labels, prediction_columns)


def run_batch_parallel(engine: DualHeadInferenceEngine, args, batch_devices: List[str]) -> Dict:
    # This function runs batch CSV inference with one worker process per GPU.
    input_csv = resolve_existing_path(args.input_csv, engine.workspace_root)
    output_path = resolve_output_path(args.output_csv, engine.workspace_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_csv(input_csv, encoding="utf-8")
    validate_batch_csv(dataframe)
    rows = dataframe.to_dict("records")
    total_rows = len(rows)
    if total_rows == 0:
        write_batch_results(output_path, [])
        return {
            "num_samples": 0,
            "input_csv": str(args.input_csv),
            "output_csv": str(args.output_csv),
            "results": [],
            "metrics": {"sample_count": 0},
        }
    if len(batch_devices) <= 1 or total_rows <= 1:
        batch_result = engine.predict_csv(
            input_csv=str(args.input_csv),
            output_csv=str(args.output_csv),
            mode=args.mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_pixels_per_image=args.pixel_budget,
            row_callback=lambda done, total, _result: print(f"[CLI] {done}/{total} completed", flush=True),
        )
        batch_result["metrics"] = summarize_batch_csv(input_csv, output_path)
        return batch_result

    print(f"[CLI] Running batch on {len(batch_devices)} GPUs: {', '.join(batch_devices)}", flush=True)
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    event_queue = ctx.Queue()
    workers = min(len(batch_devices), total_rows)
    engine_init_kwargs = {
        "workspace_root": engine.workspace_root_display,
        "registry_path": str(engine.registry_path),
        "lora_id": None if engine.model_spec.model_id == "custom" else engine.model_spec.model_id,
        "lora_path": str(engine.checkpoint_dir) if engine.model_spec.model_id == "custom" else None,
        "base_model_path": engine.base_model_path_display,
    }
    request_payload = {
        "mode": args.mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pixel_budget": args.pixel_budget if args.pixel_budget > 0 else DEFAULT_PIXEL_BUDGET,
    }
    processes = []
    for worker_id in range(workers):
        process = ctx.Process(
            target=batch_worker_loop,
            args=(
                worker_id,
                batch_devices[worker_id],
                engine_init_kwargs,
                request_payload,
                task_queue,
                event_queue,
            ),
        )
        process.start()
        processes.append(process)
    for row_index, row in enumerate(rows):
        task_queue.put(
            {
                "row_index": row_index,
                "source": str(row["source"]),
                "edited": str(row["edited"]),
                "instruction": str(row["instruction"]),
            }
        )
    for _ in range(workers):
        task_queue.put(None)
    merged_results: List[Optional[Dict]] = [None] * total_rows
    done_rows = 0
    ready_workers = 0
    last_flushed_rows = 0

    def flush_available_results():
        nonlocal last_flushed_rows
        if done_rows == last_flushed_rows:
            return
        write_batch_results(output_path, [item for item in merged_results if item is not None])
        last_flushed_rows = done_rows

    while done_rows < total_rows:
        try:
            event = event_queue.get(timeout=5.0)
        except queue.Empty:
            live_workers = sum(int(process.is_alive()) for process in processes)
            print(f"[CLI] Waiting for workers: {done_rows}/{total_rows} finished, live_workers={live_workers}", flush=True)
            continue
        if event["type"] == "worker_ready":
            ready_workers += 1
            print(f"[CLI] Worker {event['worker_id']} ready on {event['device']} ({ready_workers}/{workers})", flush=True)
            continue
        if event["type"] == "started":
            print(
                f"[CLI] Worker {event['worker_id']} on {event['device']} started row {event['row_index']}",
                flush=True,
            )
            continue
        if event["type"] == "error":
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            raise RuntimeError(f"CLI batch worker failed on {event['device']}: {event['error']}\n{event['traceback']}")
        if event["type"] == "done":
            merged_results[event["row_index"]] = event["result"]
            done_rows += 1
            print(f"[CLI] {done_rows}/{total_rows} completed across {workers} GPUs", flush=True)
            if done_rows % CSV_FLUSH_INTERVAL == 0:
                flush_available_results()
    for process in processes:
        process.join(timeout=5)
    write_batch_results(output_path, merged_results)  # type: ignore[arg-type]
    return {
        "num_samples": len(merged_results),
        "input_csv": str(args.input_csv),
        "output_csv": str(args.output_csv),
        "results": merged_results,
        "metrics": summarize_batch_csv(input_csv, output_path),
    }
