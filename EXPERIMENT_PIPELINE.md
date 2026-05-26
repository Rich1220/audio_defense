# End-to-End Experiment Pipeline

This document shows the complete workflow:

```text
benchmark
  -> target model responses
  -> safety judge labels and unsafe rate
  -> target model hidden states
  -> hidden-state router defense
```

The included fully runnable example uses:

- benchmark: `MBZUAI/AudioJailbreak`
- target model: `Qwen/Qwen2-Audio-7B-Instruct`
- judge: `meta-llama/Llama-Guard-3-8B`

For another target model, replace the target-model runner and hidden extractor
with model-specific adapters.

## 0. Install

```bash
python -m venv .venv_repro
. .venv_repro/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Choose the PyTorch command that matches your CUDA setup.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt
python -m pip install -r requirements-experiment.txt
```

Login if you need gated model access:

```bash
huggingface-cli login
```

Quick environment check:

```bash
python - <<'PY'
import torch, transformers, datasets, librosa, sklearn
print("torch", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
PY
```

## 1. Build Benchmark Manifest

AudioJailbreak smoke test:

```bash
python scripts/build_audiojailbreak_manifest.py \
  --config Origin \
  --split origin \
  --limit 20 \
  --download-audio \
  --audio-dir outputs/audiojailbreak_audio \
  --out outputs/audiojailbreak_origin_smoke20_manifest.jsonl
```

Full run:

```bash
python scripts/build_audiojailbreak_manifest.py \
  --config Origin \
  --split origin \
  --limit 0 \
  --download-audio \
  --audio-dir outputs/audiojailbreak_audio \
  --out outputs/audiojailbreak_origin_full_manifest.jsonl
```

Use `--target-model-filter TEXT` only if you have inspected the dataset and want
to keep rows whose `target_model` contains `TEXT`.

### Alternative: SACRED-Bench Multi-speaker Dialogue

SACRED-Bench is also hosted on Hugging Face and its `Multi-speaker_Dialogue`
folder is a soundfolder-style audio benchmark. Build a manifest with:

```bash
python scripts/build_hf_soundfolder_manifest.py \
  --repo-id tsinghua-ee/SACRED-Bench \
  --repo-subdir Multi-speaker_Dialogue/test \
  --limit 20 \
  --download-audio \
  --audio-dir outputs/sacred_multispeaker_audio \
  --out outputs/sacred_multispeaker_smoke20_manifest.jsonl
```

Full run:

```bash
python scripts/build_hf_soundfolder_manifest.py \
  --repo-id tsinghua-ee/SACRED-Bench \
  --repo-subdir Multi-speaker_Dialogue/test \
  --limit 0 \
  --download-audio \
  --audio-dir outputs/sacred_multispeaker_audio \
  --out outputs/sacred_multispeaker_full_manifest.jsonl
```

If the dataset metadata does not expose an original harmful prompt, the manifest
will still contain `local_audio`, but `prompt` may be empty. In that case,
Llama Guard can still judge the target model response, but prompt-conditioned
judging is weaker. Prefer metadata with a prompt/instruction field when
available.

### Alternative: JALMBench ADiv and SSJ

JALMBench is hosted as a Hugging Face `datasets` repo with named subsets. The
two subsets used here are:

```text
ADiv: 700 rows
SSJ:  246 rows
```

Build smoke manifests first:

```bash
python scripts/build_jalmbench_manifest.py \
  --subset ADiv \
  --split train \
  --limit 20 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_adiv_smoke20_manifest.jsonl

python scripts/build_jalmbench_manifest.py \
  --subset SSJ \
  --split train \
  --limit 20 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_ssj_smoke20_manifest.jsonl
```

Full manifests:

```bash
python scripts/build_jalmbench_manifest.py \
  --subset ADiv \
  --split train \
  --limit 0 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_adiv_full_manifest.jsonl

python scripts/build_jalmbench_manifest.py \
  --subset SSJ \
  --split train \
  --limit 0 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_ssj_full_manifest.jsonl
```

The manifest preserves JALMBench metadata and normalizes the text prompt to the
`prompt` field. `local_audio` points to a materialized wav file for Qwen2-Audio.

## 2. Run Target Model

Qwen2-Audio smoke run:

```bash
python scripts/run_qwen2_audio_audiojailbreak.py \
  --manifest outputs/audiojailbreak_origin_smoke20_manifest.jsonl \
  --out outputs/audiojailbreak_origin_smoke20_qwen2audio_responses.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --prompt-mode safety \
  --overwrite
```

JALMBench Qwen2-Audio full runs:

```bash
python scripts/run_qwen2_audio_audiojailbreak.py \
  --manifest outputs/jalmbench_adiv_full_manifest.jsonl \
  --out outputs/jalmbench_adiv_full_qwen2audio_responses.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --prompt-mode safety \
  --overwrite

python scripts/run_qwen2_audio_audiojailbreak.py \
  --manifest outputs/jalmbench_ssj_full_manifest.jsonl \
  --out outputs/jalmbench_ssj_full_qwen2audio_responses.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --prompt-mode safety \
  --overwrite
```

For a new model, copy:

```text
scripts/target_model_runner_template.py
```

The target-model output must preserve the manifest fields and add a response
field, for example:

```json
{"id": "example-id", "prompt": "...", "local_audio": "...", "my_model_response": "..."}
```

## 3. Judge Responses and Compute Unsafe Rate

```bash
python scripts/judge_with_llamaguard.py \
  --input outputs/audiojailbreak_origin_smoke20_qwen2audio_responses.jsonl \
  --out outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
  --response-key qwen2_audio_response \
  --prompt-key prompt \
  --model meta-llama/Llama-Guard-3-8B \
  --device cuda:0 \
  --overwrite
```

JALMBench Qwen2-Audio judge runs:

```bash
python scripts/judge_with_llamaguard.py \
  --input outputs/jalmbench_adiv_full_qwen2audio_responses.jsonl \
  --out outputs/jalmbench_adiv_full_qwen2audio_llamaguard.jsonl \
  --response-key qwen2_audio_response \
  --prompt-key prompt \
  --model meta-llama/Llama-Guard-3-8B \
  --device cuda:0 \
  --overwrite

python scripts/judge_with_llamaguard.py \
  --input outputs/jalmbench_ssj_full_qwen2audio_responses.jsonl \
  --out outputs/jalmbench_ssj_full_qwen2audio_llamaguard.jsonl \
  --response-key qwen2_audio_response \
  --prompt-key prompt \
  --model meta-llama/Llama-Guard-3-8B \
  --device cuda:0 \
  --overwrite
```

Summarize unsafe rate:

```bash
python scripts/summarize_judge_labels.py \
  --input outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
  --out-md outputs/audiojailbreak_origin_smoke20_qwen2audio_safety_summary.md
```

JALMBench summaries:

```bash
python scripts/summarize_judge_labels.py \
  --input outputs/jalmbench_adiv_full_qwen2audio_llamaguard.jsonl \
  --out-md outputs/jalmbench_adiv_full_qwen2audio_safety_summary.md

python scripts/summarize_judge_labels.py \
  --input outputs/jalmbench_ssj_full_qwen2audio_llamaguard.jsonl \
  --out-md outputs/jalmbench_ssj_full_qwen2audio_safety_summary.md
```

The router uses:

```text
llamaguard_label == "unsafe" -> label 1
everything else              -> label 0
```

You may replace Llama Guard with another judge, but normalize labels to the same
safe/unsafe convention before extracting hidden features.

## 4. Extract Hidden States

All layers:

```bash
python scripts/extract_qwen2_audio_hidden.py \
  --manifest outputs/audiojailbreak_origin_smoke20_manifest.jsonl \
  --responses outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
  --out-npz outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
  --out-meta outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0
```

JALMBench all-layer hidden extraction:

```bash
python scripts/extract_qwen2_audio_hidden.py \
  --manifest outputs/jalmbench_adiv_full_manifest.jsonl \
  --responses outputs/jalmbench_adiv_full_qwen2audio_llamaguard.jsonl \
  --out-npz outputs/jalmbench_adiv_full_qwen2audio_hidden_features.npz \
  --out-meta outputs/jalmbench_adiv_full_qwen2audio_hidden_meta.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0

python scripts/extract_qwen2_audio_hidden.py \
  --manifest outputs/jalmbench_ssj_full_manifest.jsonl \
  --responses outputs/jalmbench_ssj_full_qwen2audio_llamaguard.jsonl \
  --out-npz outputs/jalmbench_ssj_full_qwen2audio_hidden_features.npz \
  --out-meta outputs/jalmbench_ssj_full_qwen2audio_hidden_meta.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0
```

Sparse fixed layers:

```bash
python scripts/extract_qwen2_audio_hidden.py \
  --manifest outputs/audiojailbreak_origin_smoke20_manifest.jsonl \
  --responses outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
  --out-npz outputs/audiojailbreak_origin_smoke20_qwen2audio_sparse_hidden_features.npz \
  --out-meta outputs/audiojailbreak_origin_smoke20_qwen2audio_sparse_hidden_meta.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --layers 0,8,16,24,32
```

The sparse file stores `layer_indices`, so downstream reports use the real layer
ids instead of renumbering them as `0..4`.

For a new model, copy:

```text
scripts/extractor_template.py
```

and make it write:

```text
hidden_features.npz
hidden_meta.jsonl
```

See `ADAPTER_GUIDE.md` for the required schema.

## 5. Validate Features

```bash
python scripts/validate_features.py \
  --features outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
  --meta outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl
```

## 6. Run Defense

For smoke tests, use `SPLIT_MODE=random` because held-out source/category splits
may be too small:

```bash
FEATURES=outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
META=outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/audiojailbreak_origin_smoke20_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=random \
OBJECTIVE=f1 \
bash run_hidden_router_pipeline.sh
```

For full runs, use all split types:

```bash
FEATURES=outputs/audiojailbreak_origin_full_qwen2audio_hidden_features.npz \
META=outputs/audiojailbreak_origin_full_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/audiojailbreak_origin_full_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=all \
OBJECTIVE=f1 \
bash run_hidden_router_pipeline.sh
```

For JALMBench full runs, use random split first. ADiv and SSJ are each one
subset/attack type, so source/category held-out splits may not be meaningful.

```bash
FEATURES=outputs/jalmbench_adiv_full_qwen2audio_hidden_features.npz \
META=outputs/jalmbench_adiv_full_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/jalmbench_adiv_full_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=random \
OBJECTIVE=high_recall \
bash run_hidden_router_pipeline.sh

FEATURES=outputs/jalmbench_ssj_full_qwen2audio_hidden_features.npz \
META=outputs/jalmbench_ssj_full_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/jalmbench_ssj_full_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=random \
OBJECTIVE=high_recall \
bash run_hidden_router_pipeline.sh
```

## 7. Read Results

Start with:

```text
OUT_DIR/single_layer_probe/probe_summary.md
OUT_DIR/single_router_defense/defense_simulation_summary.md
OUT_DIR/auto_layer_router/auto_layer_router_summary.md
```

Report:

- benchmark and target model
- judge model
- number of valid examples
- unsafe rate before defense
- best AUROC/AUPRC
- route threshold objective
- unsafe rate after defense simulation
- route rate
- false-route rate on safe examples
- held-out source/category results when available

## 8. Optional Category-Transfer Analysis

This analysis is different from `SPLIT_MODE=category`.

```text
SPLIT_MODE=category:
  train = all categories except one
  test  = the held-out category

category transfer:
  train = one category
  test  = every eligible category
```

Run it after hidden features have been extracted:

```bash
python scripts/train_category_transfer_router.py \
  --features outputs/audiojailbreak_origin_full_qwen2audio_hidden_features.npz \
  --meta outputs/audiojailbreak_origin_full_qwen2audio_hidden_meta.jsonl \
  --out-dir outputs/audiojailbreak_origin_full_qwen2audio_category_transfer
```

For small smoke tests, lower the positive-count requirements:

```bash
python scripts/train_category_transfer_router.py \
  --features outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
  --meta outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl \
  --out-dir outputs/audiojailbreak_origin_smoke20_qwen2audio_category_transfer \
  --min-train-positives 1 \
  --min-test-positives 1
```

Main outputs:

```text
category_transfer_summary.md
category_transfer_results.csv
category_transfer_selected_layers.json
category_transfer_auroc_heatmap.png
category_transfer_recall_heatmap.png
category_transfer_budget_recall_heatmap.png
```

## What Is Model-Specific

These steps are generic and included:

- benchmark manifest creation
- Llama Guard judging
- unsafe-rate summary
- router training and defense simulation

These steps are model-specific:

- target model response generation
- hidden-state extraction

This repo includes Qwen2-Audio versions of those model-specific steps and
templates for other models.
