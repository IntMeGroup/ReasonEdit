# This module provides the ReasonEdit CLI for single-sample and batch inference.

import argparse
import json
from pathlib import Path
from typing import List

try:
    from ReasonEdit.core.inference import DEFAULT_PIXEL_BUDGET, DualHeadInferenceEngine, run_batch_parallel
except ImportError:  # pragma: no cover - supports running from the repo root.
    from core.inference import DEFAULT_PIXEL_BUDGET, DualHeadInferenceEngine, run_batch_parallel


def parse_args():
    # This function parses CLI arguments for ReasonEdit inference.
    parser = argparse.ArgumentParser(description="ReasonEdit image edit evaluator CLI.")
    parser.add_argument("--workspace_root", default=".")
    parser.add_argument("--registry_path", default="config/models.json")
    parser.add_argument("--base_model_path", default="")
    parser.add_argument("--lora_id", default="reasonedit_v1")
    parser.add_argument("--lora_path", default="")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--devices", default="")
    parser.add_argument("--mode", choices=["score_only", "cot", "with_cot"], default="cot")
    parser.add_argument("--source_image", default="")
    parser.add_argument("--edited_image", default="")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--input_csv", default="")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--pixel_budget", type=int, default=DEFAULT_PIXEL_BUDGET)
    parser.add_argument("--print_config", action="store_true")
    return parser.parse_args()


def parse_device_list(device: str, devices_text: str, gpu_id: int) -> List[str]:
    # This function parses one primary device plus an optional comma-separated device list.
    raw = [item.strip() for item in str(devices_text or "").split(",") if item.strip()]
    if raw:
        return raw
    if str(device or "").strip():
        return [str(device).strip()]
    return [f"cuda:{gpu_id}"]


def main():
    # This function dispatches the requested CLI inference flow.
    args = parse_args()
    batch_devices = parse_device_list(args.device, args.devices, args.gpu_id)
    engine = DualHeadInferenceEngine(
        workspace_root=args.workspace_root,
        registry_path=args.registry_path,
        lora_id=args.lora_id or None,
        lora_path=args.lora_path or None,
        base_model_path=args.base_model_path or None,
        device=batch_devices[0],
        load_on_init=False,
    )
    if args.print_config:
        payload = engine.describe()
        payload["batch_devices"] = batch_devices
        payload["batch_parallel_enabled"] = len(batch_devices) > 1
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return
    if args.input_csv:
        if not args.output_csv:
            default_name = Path(args.input_csv).stem + "_predictions.csv"
            args.output_csv = str(Path("outputs") / default_name)
        result = run_batch_parallel(engine, args, batch_devices=batch_devices)
        print(
            json.dumps(
                {
                    "num_samples": result["num_samples"],
                    "output_csv": result["output_csv"],
                    "batch_devices": batch_devices,
                    "batch_parallel_enabled": len(batch_devices) > 1,
                    "metrics": result.get("metrics", {}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    if not args.source_image or not args.edited_image or not args.instruction:
        raise ValueError("Single-sample inference requires --source_image, --edited_image, and --instruction")
    result = engine.predict(
        source_path=args.source_image,
        edited_path=args.edited_image,
        instruction=args.instruction,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_pixels_per_image=args.pixel_budget,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
