<h1 align="center">
ReasonEdit: Towards Interpretable Image Editing Evaluation via Reinforcement Learning
</h1>

## 📦 Install

```bash
python -m pip install -r requirements.txt
```

## 📥 Model Weights
Download the model weights from the [link](https://huggingface.co/TmpAccount/ReasonEdit/tree/main) and put them into:

```text
weights/qwen35_9b
weights/qwen35_4b
weights/ReasonEdit
weights/RE-Reward
```

You can override the defaults at launch with `--base_model_path` and `--lora_path`.

The default registry points to:

```text
base model: weights/qwen35_9b
LoRA weights: weights/ReasonEdit
```
## ⚡Inference
### CLI Inference

Single sample, default CoT mode:

```bash
python -m cli.infer_cli \
  --source_image example/example_source.jpg \
  --edited_image example/example_edited.jpg \
  --instruction "$(cat example/example_instruction.txt)"
```

Score-only mode:

```bash
python -m cli.infer_cli \
  --mode score_only \
  --source_image example/example_source.jpg \
  --edited_image example/example_edited.jpg \
  --instruction "$(cat example/example_instruction.txt)"
```

Batch CSV:

```bash
python -m cli.infer_cli \
  --input_csv test_samples/official_noleak_test/test.csv \
  --output_csv outputs/reasonedit_predictions.csv
```

The batch CSV must contain `source`, `edited`, and `instruction` columns. Image paths in the CSV are resolved relative to this repository root when they are not absolute.
The output CSV keeps only `source`, `edited`, `instruction`, `v_score`, `e_score`, `p_score`, and `cot_text`. When the input CSV also carries labels, the CLI prints three-dimensional `PLCC`, `SRCC`, and `KRCC` metrics in its JSON summary.

### Multi-GPU Batch

```bash
python -m cli.infer_cli \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --mode score_only \
  --input_csv test_samples/official_noleak_test/test.csv \
  --output_csv outputs/reasonedit_predictions.csv
```

### Reward Model Package

The reward model release is split into code and weights:

```text
RE-reward/
weights/RE-Reward/
```

Single-sample inference:

```bash
python RE-reward/reward_infer.py sample \
  --image_0 example/example_source.jpg \
  --image_1 example/example_edited.jpg \
  --instruction "$(cat example/example_instruction.txt)" \
  --critique "$(cat example/reward_example_critique.txt)"
```

CSV batch inference:

```bash
python RE-reward/reward_infer.py csv \
  --csv_path test_samples/rm_test_stratified_2000/test_2000_sft_like.csv \
  --output_csv outputs/rm_predictions.csv
```

The CSV can use either `image_0`/`image_1`/`critique` or `source`/`edited`/`cot` columns. If labels are present, the script reports three-dimensional `PLCC`, `SRCC`, and `KRCC` metrics. The default base model path is `weights/qwen35_4b`, and the default LoRA package path is `weights/RE-Reward`.

`--pixel_budget` defaults to `262144`. `calibration.json` is optional; when it is absent, scores use identity calibration. The final reward is `0.3 * visual_quality + 0.4 * editing_alignment + 0.3 * content_preservation`.
