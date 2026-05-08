# This module loads a single-GPU Qwen3.5-4B LoRA reward model and exposes callable reward functions.

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from scipy.stats import kendalltau

SCORE_COLS = ["logicality_score", "accuracy_score", "usefulness_score"]
DISPLAY_SCORE_COLS = ["logicality", "accuracy", "usefulness"]
LABEL_COLUMN_ALIASES = (
    ("logicality_score", "accuracy_score", "usefulness_score"),
    ("v_score", "e_score", "p_score"),
)
REWARD_WEIGHTS = {
    "logicality": 0.3,
    "accuracy": 0.4,
    "usefulness": 0.3,
}
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL_PATH = str(WORKSPACE_ROOT / "weights" / "qwen35_4b")
DEFAULT_PACKAGE_DIR = str(WORKSPACE_ROOT / "weights" / "RE-Reward")


class RewardHead(torch.nn.Module):
    # This module maps the pooled Qwen hidden state to three reward dimensions.
    def __init__(self, hidden_size: int, bottleneck: int, dropout: float, num_labels: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, bottleneck),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(bottleneck, num_labels),
        )

    # This method applies the reward head to one pooled hidden-state tensor.
    def forward(self, pooled_hidden_state: torch.Tensor) -> torch.Tensor:
        return self.layers(pooled_hidden_state)

    # This method exposes a flat state dict compatible with the saved training-time key layout.
    def load_state_dict(self, state_dict, strict: bool = True):
        if "0.weight" in state_dict:
            remapped_state_dict = {f"layers.{key}": value for key, value in state_dict.items()}
            return super().load_state_dict(remapped_state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)


class CsvRewardDataset(Dataset):
    # This dataset converts a reward-model CSV into single-GPU inference batches.
    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_wrapper,
        project_root: str,
        max_length: int,
        max_pixels_per_image: int,
        label_columns=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.model_wrapper = model_wrapper
        self.project_root = project_root
        self.max_length = int(max_length)
        self.max_pixels_per_image = int(max_pixels_per_image)
        self.label_columns = list(label_columns or [])

    # This method returns the row count.
    def __len__(self):
        return len(self.dataframe)

    # This method truncates critique text to fit the configured token budget.
    def _truncate_critique_to_fit(self, critique_text: str, overflow_tokens: int) -> str:
        critique_token_ids = self.model_wrapper.processor.tokenizer(critique_text, add_special_tokens=False)["input_ids"]
        keep_tokens = max(0, len(critique_token_ids) - overflow_tokens - 32)
        if keep_tokens == 0:
            raise ValueError("Critique text does not fit within max_length after token-budget truncation")
        return self.model_wrapper.processor.tokenizer.decode(
            critique_token_ids[:keep_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    # This method returns one encoded reward-model sample.
    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        source_image = load_rgb_image(resolve_image_path(self.project_root, row["image_0"]), self.max_pixels_per_image)
        edited_image = load_rgb_image(resolve_image_path(self.project_root, row["image_1"]), self.max_pixels_per_image)
        critique_text = str(row["critique"]).strip()
        encoded = self.model_wrapper.encode(source_image, edited_image, str(row["instruction"]).strip(), critique_text)
        sequence_length = int(encoded["input_ids"].shape[1])
        if sequence_length > self.max_length:
            critique_text = self._truncate_critique_to_fit(critique_text, sequence_length - self.max_length)
            encoded = self.model_wrapper.encode(source_image, edited_image, str(row["instruction"]).strip(), critique_text)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        mm_token_type_ids = encoded["mm_token_type_ids"].squeeze(0)
        if input_ids.shape[0] > self.max_length:
            raise ValueError(f"Encoded sequence length {input_ids.shape[0]} exceeds max_length {self.max_length}")
        payload = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
        }
        if hasattr(self, "label_columns") and self.label_columns:
            payload["labels"] = torch.tensor([float(row[column]) for column in self.label_columns], dtype=torch.float32)
        return payload


class Qwen35LoRARewardModel(torch.nn.Module):
    # This class wraps the base model, LoRA adapter, and reward head for single-GPU scoring.
    def __init__(self, base_model: str, package_dir: str, precision: str = "bf16", device: str = "cuda:0"):
        super().__init__()
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in dtype_map:
            raise ValueError(f"Unsupported precision: {precision}")

        self.base_model = str(base_model)
        self.package_dir = str(Path(package_dir).resolve())
        self.device = torch.device(device)
        self.dtype = dtype_map[precision]

        self.processor = AutoProcessor.from_pretrained(self.base_model, trust_remote_code=True)
        self.vl_model = AutoModelForImageTextToText.from_pretrained(
            self.base_model,
            dtype=self.dtype,
            trust_remote_code=True,
        )
        self.vl_model = PeftModel.from_pretrained(
            self.vl_model,
            self.package_dir,
            is_trainable=False,
        )
        if not hasattr(self.vl_model, "get_base_model"):
            raise AttributeError("LoRA model does not expose get_base_model")

        reward_head_config_path = Path(self.package_dir) / "reward_head_config.json"
        with open(reward_head_config_path, "r", encoding="utf-8") as handle:
            reward_head_config = json.load(handle)
        self.reward_head_config = reward_head_config
        self.score_head = RewardHead(
            hidden_size=int(reward_head_config["hidden_size"]),
            bottleneck=int(reward_head_config["bottleneck"]),
            dropout=float(reward_head_config["dropout"]),
            num_labels=int(reward_head_config["num_labels"]),
        )
        score_head_path = Path(self.package_dir) / "score_head.pth"
        self.score_head.load_state_dict(torch.load(score_head_path, map_location="cpu", weights_only=True))
        self.vl_model = self.vl_model.to(self.device)
        self.score_head = self.score_head.to(self.device)
        self.vl_model.eval()
        self.score_head.eval()

    # This method builds the reward-model prompt.
    def build_prompt(self, instruction: str, critique_text: str) -> str:
        return (
            "Task: evaluate the candidate critique for an image editing example.\n"
            "Image-1 is the original source image and Image-2 is the edited result image.\n"
            f"Editing instruction:\n{instruction.strip()}\n\n"
            "Candidate critique:\n"
            f"{critique_text.strip()}\n\n"
            "Judge three reward dimensions jointly:\n"
            "1. logicality: internal consistency, coherent reasoning, and absence of contradictions.\n"
            "2. accuracy: factual alignment with the source image, edited image, and editing instruction.\n"
            "3. usefulness: specificity, diagnostic value, and usefulness for reward modeling.\n"
            "Summarize the grounded evidence into the final anchor token sequence for regression.\n"
            "Reward Scores:"
        )

    # This method tokenizes one multimodal reward sample.
    def encode(self, source_image, edited_image, instruction: str, critique_text: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": source_image},
                    {"type": "image", "image": edited_image},
                    {"type": "text", "text": self.build_prompt(instruction, critique_text)},
                ],
            }
        ]
        payload = self.processor.apply_chat_template(
            messages,
            chat_template=self.processor.tokenizer.chat_template,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        required_keys = ["pixel_values", "image_grid_thw", "mm_token_type_ids"]
        missing_keys = [key for key in required_keys if payload.get(key) is None]
        if missing_keys:
            raise ValueError(f"Processor output is missing required multimodal keys: {missing_keys}")
        return payload

    # This method applies the reward model to one padded batch.
    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids):
        backbone = self.vl_model.get_base_model().model
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise ValueError("Backbone forward did not return last_hidden_state")
        seq_lens = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_idx, seq_lens].to(self.score_head.layers[0].weight.dtype)
        return self.score_head(pooled)

    # This method computes one reward dict from raw inputs.
    def score(self, image_0: str, image_1: str, instruction: str, critique: str, max_pixels_per_image: int = 0):
        source_image = load_rgb_image(str(image_0), max_pixels_per_image)
        edited_image = load_rgb_image(str(image_1), max_pixels_per_image)
        payload = self.encode(source_image, edited_image, instruction, critique)
        with torch.inference_mode():
            logits = self(
                input_ids=payload["input_ids"].to(self.device),
                attention_mask=payload["attention_mask"].to(self.device),
                pixel_values=payload["pixel_values"].to(self.device),
                image_grid_thw=payload["image_grid_thw"].to(self.device),
                mm_token_type_ids=payload["mm_token_type_ids"].to(self.device),
            )
        values = logits[0].float().cpu().tolist()
        return {name: float(value) for name, value in zip(DISPLAY_SCORE_COLS, values)}


# This function creates a loaded single-GPU reward model instance.
def load_reward_model(base_model: str, package_dir: str, precision: str = "bf16", device: str = "cuda:0") -> Qwen35LoRARewardModel:
    return Qwen35LoRARewardModel(base_model=base_model, package_dir=package_dir, precision=precision, device=device)


# This function converts three reward dimensions into one weighted final reward.
def aggregate_reward_scores(scores):
    return float(sum(float(scores[name]) * float(weight) for name, weight in REWARD_WEIGHTS.items()))


# This function computes one reward dict from raw multimodal inputs.
def compute_reward(model: Qwen35LoRARewardModel, image_0: str, image_1: str, instruction: str, critique: str, max_pixels_per_image: int = 0):
    scores = model.score(
        image_0=image_0,
        image_1=image_1,
        instruction=instruction,
        critique=critique,
        max_pixels_per_image=max_pixels_per_image,
    )
    scores["reward"] = aggregate_reward_scores(scores)
    return scores


# This function builds the collate function for single-GPU batch scoring.
def make_collate_fn(pad_token_id: int):
    # This nested function pads variable-length batch items.
    def collate_fn(batch):
        output = {
            "input_ids": pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0),
            "mm_token_type_ids": pad_sequence([item["mm_token_type_ids"] for item in batch], batch_first=True, padding_value=0),
            "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
        }
        if "labels" in batch[0]:
            output["labels"] = torch.stack([item["labels"] for item in batch])
        return output

    return collate_fn


# This function loads one RGB image and optionally resizes it to a pixel budget.
def load_rgb_image(image_path: str, max_pixels_per_image: int):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        pixels = width * height
        if max_pixels_per_image <= 0 or pixels <= max_pixels_per_image:
            return image
        scale = math.sqrt(max_pixels_per_image / float(pixels))
        new_width = max(16, int(round(width * scale)))
        new_height = max(16, int(round(height * scale)))
        new_width = max(16, (new_width // 16) * 16)
        new_height = max(16, (new_height // 16) * 16)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# This function reads one CSV with strict UTF-8 decoding.
def read_csv_strict(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8")


# This function validates the minimum inference CSV schema.
def validate_columns(dataframe: pd.DataFrame):
    if {"image_0", "image_1", "instruction", "critique"}.issubset(dataframe.columns):
        return
    if {"source", "edited", "instruction", "cot"}.issubset(dataframe.columns):
        return
    raise ValueError(
        "Missing required columns. Expected either image_0/image_1/instruction/critique or "
        "source/edited/instruction/cot."
    )


def normalize_dataframe_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    # This function normalizes supported batch CSV schemas into one internal layout.
    if {"image_0", "image_1", "instruction", "critique"}.issubset(dataframe.columns):
        return dataframe.copy()
    if {"source", "edited", "instruction", "cot"}.issubset(dataframe.columns):
        return dataframe.rename(columns={"source": "image_0", "edited": "image_1", "cot": "critique"}).copy()
    raise ValueError(
        "Unsupported CSV schema. Expected either image_0/image_1/instruction/critique or "
        "source/edited/instruction/cot."
    )


def select_label_columns(dataframe: pd.DataFrame):
    # This function selects whichever three label columns the CSV already provides.
    for candidate in LABEL_COLUMN_ALIASES:
        if set(candidate).issubset(dataframe.columns):
            return list(candidate)
    return None


def safe_corr_np(x: np.ndarray, y: np.ndarray):
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0, 0.0, 0.0
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
    return plcc, srcc, krcc


def summarize_predictions(predictions: np.ndarray, labels: np.ndarray):
    metrics = {"sample_count": int(predictions.shape[0])}
    for index, score_name in enumerate(DISPLAY_SCORE_COLS):
        plcc, srcc, krcc = safe_corr_np(predictions[:, index], labels[:, index])
        metrics[f"{score_name}_plcc"] = float(plcc)
        metrics[f"{score_name}_srcc"] = float(srcc)
        metrics[f"{score_name}_krcc"] = float(krcc)
    return metrics


# This function resolves one CSV image path relative to a project root.
def resolve_image_path(project_root: str, raw_path: str) -> str:
    normalized = str(raw_path).replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        resolved = path
    else:
        resolved = Path(project_root) / normalized
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")
    return str(resolved)


# This function runs batch scoring from one CSV on a single GPU.
def score_csv(model: Qwen35LoRARewardModel, csv_path: str, project_root: str, batch_size: int, num_workers: int, prefetch_factor: int, persistent_workers: bool, max_length: int, max_pixels_per_image: int):
    dataframe = normalize_dataframe_schema(read_csv_strict(csv_path))
    validate_columns(dataframe)
    label_columns = select_label_columns(dataframe)
    has_labels = label_columns is not None
    if has_labels:
        for column in label_columns:
            dataframe[column] = dataframe[column].astype(float)

    dataset = CsvRewardDataset(
        dataframe=dataframe,
        model_wrapper=model,
        project_root=project_root,
        max_length=max_length,
        max_pixels_per_image=max_pixels_per_image,
        label_columns=label_columns,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": None,
        "collate_fn": make_collate_fn(model.processor.tokenizer.pad_token_id),
        "num_workers": num_workers,
        "pin_memory": model.device.type == "cuda",
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    data_loader = DataLoader(**loader_kwargs)

    predictions = []
    labels = []
    use_amp = model.device.type == "cuda" and model.dtype in {torch.float16, torch.bfloat16}

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="score_csv"):
            input_ids = batch["input_ids"].to(model.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(model.device, non_blocking=True)
            image_grid_thw = batch["image_grid_thw"].to(model.device, non_blocking=True)
            mm_token_type_ids = batch["mm_token_type_ids"].to(model.device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=model.dtype):
                    batch_predictions = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        mm_token_type_ids=mm_token_type_ids,
                    )
            else:
                batch_predictions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                )
            predictions.append(batch_predictions.float().cpu())
            if "labels" in batch:
                labels.append(batch["labels"].float().cpu())

    prediction_array = torch.cat(predictions, dim=0).numpy()
    result_df = dataframe.copy()
    for index, score_name in enumerate(DISPLAY_SCORE_COLS):
        result_df[f"pred_{score_name}"] = prediction_array[:, index]
    result_df["pred_reward"] = prediction_array @ np.array([REWARD_WEIGHTS[name] for name in DISPLAY_SCORE_COLS], dtype=np.float32)

    metrics = {
        "sample_count": int(len(result_df)),
    }
    if labels:
        label_array = torch.cat(labels, dim=0).numpy()
        metrics.update(summarize_predictions(prediction_array, label_array))
    return result_df, metrics


# This function builds the CLI parser.
def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL_PATH)
    sample_parser.add_argument("--package_dir", type=str, default=DEFAULT_PACKAGE_DIR)
    sample_parser.add_argument("--image_0", type=str, required=True)
    sample_parser.add_argument("--image_1", type=str, required=True)
    sample_parser.add_argument("--instruction", type=str, required=True)
    sample_parser.add_argument("--critique", type=str, required=True)
    sample_parser.add_argument("--max_pixels_per_image", type=int, default=0)
    sample_parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    sample_parser.add_argument("--device", type=str, default="cuda:0")
    sample_parser.add_argument("--output_json", type=str, default="")

    csv_parser = subparsers.add_parser("csv")
    csv_parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL_PATH)
    csv_parser.add_argument("--package_dir", type=str, default=DEFAULT_PACKAGE_DIR)
    csv_parser.add_argument("--csv_path", type=str, required=True)
    csv_parser.add_argument("--project_root", type=str, default="")
    csv_parser.add_argument("--batch_size", type=int, default=1)
    csv_parser.add_argument("--num_workers", type=int, default=2)
    csv_parser.add_argument("--prefetch_factor", type=int, default=2)
    csv_parser.add_argument("--persistent_workers", action="store_true")
    csv_parser.add_argument("--max_length", type=int, default=24576)
    csv_parser.add_argument("--max_pixels_per_image", type=int, default=0)
    csv_parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    csv_parser.add_argument("--device", type=str, default="cuda:0")
    csv_parser.add_argument("--output_csv", type=str, default="")
    csv_parser.add_argument("--metrics_json", type=str, default="")
    return parser


# This function runs one single-sample CLI call.
def run_sample_command(args):
    model = load_reward_model(
        base_model=args.base_model,
        package_dir=args.package_dir,
        precision=args.precision,
        device=args.device,
    )
    scores = compute_reward(
        model=model,
        image_0=str(Path(args.image_0).resolve()),
        image_1=str(Path(args.image_1).resolve()),
        instruction=args.instruction,
        critique=args.critique,
        max_pixels_per_image=args.max_pixels_per_image,
    )
    payload = {
        "base_model": args.base_model,
        "package_dir": str(Path(args.package_dir).resolve()),
        "image_0": str(Path(args.image_0).resolve()),
        "image_1": str(Path(args.image_1).resolve()),
        "scores": scores,
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


# This function runs one CSV CLI call.
def run_csv_command(args):
    project_root = str(Path(args.project_root).resolve()) if args.project_root else str(Path.cwd())
    model = load_reward_model(
        base_model=args.base_model,
        package_dir=args.package_dir,
        precision=args.precision,
        device=args.device,
    )
    result_df, metrics = score_csv(
        model=model,
        csv_path=args.csv_path,
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        max_length=args.max_length,
        max_pixels_per_image=args.max_pixels_per_image,
    )
    metrics.update(
        {
            "base_model": args.base_model,
            "package_dir": str(Path(args.package_dir).resolve()),
            "csv_path": str(Path(args.csv_path).resolve()),
        }
    )
    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False, encoding="utf-8")
        metrics["output_csv"] = str(output_csv)
    if args.metrics_json:
        metrics_json = Path(args.metrics_json).resolve()
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        metrics["metrics_json"] = str(metrics_json)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


# This function parses CLI arguments and dispatches one single-GPU command.
def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "sample":
        run_sample_command(args)
        return
    if args.command == "csv":
        run_csv_command(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
