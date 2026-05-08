# This module holds prompt and CSV validation utilities.

from pathlib import Path

import pandas as pd

DEFAULT_SCORE_ANCHOR = "[Final Assessment]"
RECOMMENDED_BASE_MODEL_PATH = "weights/qwen35_9b"
IMAGE_SUFFIX_CANDIDATES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_prompt(instruction: str) -> str:
    # This function builds the multimodal evaluation prompt.
    return (
        "You are an expert in image editing quality assessment, specializing in evaluating AI-edited images. "
        "Please output the evaluation in the most concise manner.\n\n"
        "Original Image: \\img1\n"
        "Edited Image: \\img2\n"
        f"Editing Instruction: {str(instruction).strip()}\n\n"
        "Evaluation Framework:\n"
        "[Original Image Description]\n"
        "What is the content of the original image?\n\n"
        "[Edited Image Description]\n"
        "What is the content of the edited image?\n\n"
        "[Evaluation Rationale] Evaluate based on the following three dimensions, providing at least two points for each.\n"
        "1. Visual Quality (Naturalness of the edit and image)\n"
        "E.g., lighting, clarity, color, details, realism, etc.\n"
        "2. Editing Alignment (Adherence to editing instructions)\n"
        "Whether the instruction is fully or partially implemented, and the effectiveness of the implementation.\n"
        "3. Content Preservation (Content consistency)\n"
        "E.g., consistency of the main structure with the original, preservation of unedited areas, style consistency.\n\n"
        "[Final Assessment]\n"
        "After outputting [Final Assessment], immediately continue with exactly three scores for Visual Quality, Editing Alignment, "
        "and Content Preservation in one line, separated by commas, with no extra words.\n"
        "Use the format X.XX, X.XX, X.XX with exactly two decimal places.\n"
        "Example: [Final Assessment]0.58, 0.36, 0.50"
    )


def validate_batch_csv(dataframe: pd.DataFrame):
    # This function validates the required columns for batch inference.
    required = {"source", "edited", "instruction"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def resolve_existing_path(path_text: str, workspace_root: str) -> Path:
    # This function resolves one user-provided local path and fails if missing.
    raw = Path(str(path_text).strip())
    candidate = raw if raw.is_absolute() else Path(workspace_root) / raw
    candidate = candidate.resolve()
    if candidate.exists():
        return candidate
    if candidate.suffix:
        raise FileNotFoundError(f"Path not found: {candidate}")
    tried = [str(candidate)]
    for suffix in IMAGE_SUFFIX_CANDIDATES:
        candidate_with_suffix = Path(str(candidate) + suffix)
        tried.append(str(candidate_with_suffix))
        if candidate_with_suffix.exists():
            return candidate_with_suffix.resolve()
    raise FileNotFoundError(f"Path not found. Tried: {', '.join(tried)}")
