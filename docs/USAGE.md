# Usage Guide

This guide shows how to run the hidden-state router defense on supported
benchmarks.

Supported benchmark sources:

```text
AudioJailbreak
JALMBench ADiv / SSJ
SACRED-Bench Multi-speaker Dialogue
Generic Hugging Face soundfolder-style datasets
```

## 1. Setup

From the repository root:

```bash
cd /path/to/defense_method
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -r requirements-experiment.txt
```

Check the environment:

```bash
python - <<'PY'
import torch, transformers, datasets, librosa, sklearn, numpy
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
PY
```

If you need a specific GPU, configure it according to your machine. For example:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Then use `--device cuda:0` inside commands. When `CUDA_VISIBLE_DEVICES` is set,
`cuda:0` means the first visible GPU, not necessarily physical GPU 0.

If you need gated model access:

```bash
huggingface-cli login
```

## 2. Naming Convention

Use a stable run prefix:

```text
{benchmark}_{subset}_{size}_qwen2audio
```

Examples:

```text
audiojailbreak_origin_smoke20_qwen2audio
jalmbench_adiv_p100_qwen2audio
jalmbench_ssj_full_qwen2audio
sacred_multispeaker_full_qwen2audio
```

The commands below follow this naming style.

## 3. AudioJailbreak

### Build Manifest

Smoke test:

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

### Run Qwen2-Audio

```bash
python scripts/run_qwen2_audio_audiojailbreak.py \
  --manifest outputs/audiojailbreak_origin_smoke20_manifest.jsonl \
  --out outputs/audiojailbreak_origin_smoke20_qwen2audio_responses.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --prompt-mode safety \
  --overwrite
```

### Judge Responses

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

Summarize labels:

```bash
python scripts/summarize_judge_labels.py \
  --input outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
  --out-md outputs/audiojailbreak_origin_smoke20_qwen2audio_safety_summary.md
```

### Extract Hidden States

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

Sparse layers:

```bash
python scripts/extract_qwen2_audio_hidden.py \
  --manifest outputs/audiojailbreak_origin_smoke20_manifest.jsonl \
  --responses outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
  --out-npz outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
  --out-meta outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --layers 0,8,16,24,32
```

### Run Defense

```bash
FEATURES=outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
META=outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/audiojailbreak_origin_smoke20_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=random \
OBJECTIVE=f1 \
bash run_hidden_router_pipeline.sh
```

For full runs, use:

```bash
SPLIT_MODE=all
```

## 4. JALMBench

JALMBench currently supports:

```text
ADiv
SSJ
```

### Build Manifests

Recommended combined smoke test. This writes ADiv and SSJ into one manifest:

```bash
python scripts/build_jalmbench_manifest.py \
  --subset all \
  --split train \
  --limit 20 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_all_smoke20_manifest.jsonl
```

`--limit` is applied per subset, so `--subset all --limit 20` writes up to 40
rows total.

Combined full run:

```bash
python scripts/build_jalmbench_manifest.py \
  --subset all \
  --split train \
  --limit 0 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_all_full_manifest.jsonl
```

You can still build subsets separately:

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
python scripts/build_jalmbench_manifest.py \
  --subset SSJ \
  --split train \
  --limit 20 \
  --save-audio \
  --audio-dir outputs/jalmbench_audio \
  --out outputs/jalmbench_ssj_smoke20_manifest.jsonl
```

### Run Qwen2-Audio

```bash
python scripts/run_qwen2_audio_audiojailbreak.py \
  --manifest outputs/jalmbench_all_smoke20_manifest.jsonl \
  --out outputs/jalmbench_all_smoke20_qwen2audio_responses.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --prompt-mode safety \
  --overwrite
```

### Judge Responses

```bash
python scripts/judge_with_llamaguard.py \
  --input outputs/jalmbench_all_smoke20_qwen2audio_responses.jsonl \
  --out outputs/jalmbench_all_smoke20_qwen2audio_llamaguard.jsonl \
  --response-key qwen2_audio_response \
  --prompt-key prompt \
  --model meta-llama/Llama-Guard-3-8B \
  --device cuda:0 \
  --overwrite
```

Summarize labels:

```bash
python scripts/summarize_judge_labels.py \
  --input outputs/jalmbench_all_smoke20_qwen2audio_llamaguard.jsonl \
  --out-md outputs/jalmbench_all_smoke20_qwen2audio_safety_summary.md
```

If all labels are safe, the binary router cannot train. Increase the limit, for
example `--limit 100`, or run the full subset.

### Extract Hidden States

```bash
python scripts/extract_qwen2_audio_hidden.py \
  --manifest outputs/jalmbench_all_smoke20_manifest.jsonl \
  --responses outputs/jalmbench_all_smoke20_qwen2audio_llamaguard.jsonl \
  --out-npz outputs/jalmbench_all_smoke20_qwen2audio_hidden_features.npz \
  --out-meta outputs/jalmbench_all_smoke20_qwen2audio_hidden_meta.jsonl \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0
```

### Run Defense

JALMBench subsets are usually single-source/single-category runs, so start with
random split:

```bash
FEATURES=outputs/jalmbench_all_smoke20_qwen2audio_hidden_features.npz \
META=outputs/jalmbench_all_smoke20_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/jalmbench_all_smoke20_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=all \
OBJECTIVE=high_recall \
bash run_hidden_router_pipeline.sh
```

## 5. SACRED-Bench

SACRED-Bench can be loaded through the generic Hugging Face soundfolder manifest
builder.

### Build Manifest

Smoke test:

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

Then run Qwen2-Audio, judge, hidden extraction, and defense using the same
pattern as above with the `sacred_multispeaker_*` prefix.

## 6. Validate Feature Files

Before training:

```bash
python scripts/validate_features.py \
  --features outputs/{run_name}_qwen2audio_hidden_features.npz \
  --meta outputs/{run_name}_qwen2audio_hidden_meta.jsonl
```

Expected:

```text
[OK] feature schema is valid
[INFO] labels: {0: ..., 1: ...}
```

If labels contain only one class, the router cannot train.

## 7. Read Results

Start with:

```bash
cat outputs/{run_name}_qwen2audio_safety_summary.md
cat outputs/{run_name}_qwen2audio_defense/single_layer_probe/probe_summary.md
cat outputs/{run_name}_qwen2audio_defense/auto_layer_router/auto_layer_router_summary.md
```

Important fields:

| Field | Meaning |
|---|---|
| `AUROC` | Ranking quality of unsafe risk scores. |
| `AUPRC` | Precision-recall quality for unsafe examples. |
| `Unsafe Before` | Unsafe rate before routing. |
| `Unsafe After` | Residual unsafe rate after simulated routing. |
| `Reduction` | Fraction of unsafe examples caught by routing. |
| `Route Rate` | Fraction of all examples routed to the safer path. |
| `False Route` | Fraction of safe examples incorrectly routed. |
| `Recall` | Unsafe caught recall. |
| `Threshold` | Validation-selected routing threshold. |

## 8. Phase 1 Interpretability on AudioJailbreak

Phase 1 adds attention-deviation analysis on top of the hidden router. Start
from an AudioJailbreak run that already has:

```text
outputs/audiojailbreak_origin_smoke20_manifest.jsonl
outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl
outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz
outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl
```

Run the hidden router if you have not already:

```bash
FEATURES=outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_features.npz \
META=outputs/audiojailbreak_origin_smoke20_qwen2audio_hidden_meta.jsonl \
OUT_DIR=outputs/audiojailbreak_origin_smoke20_qwen2audio_defense \
PYTHON=python \
SPLIT_MODE=random \
OBJECTIVE=f1 \
bash run_hidden_router_pipeline.sh
```

Extract attention-span features:

```bash
python scripts/extract_qwen2_audio_attention.py \
  --manifest outputs/audiojailbreak_origin_smoke20_manifest.jsonl \
  --responses outputs/audiojailbreak_origin_smoke20_qwen2audio_llamaguard.jsonl \
  --out-jsonl outputs/audiojailbreak_origin_smoke20_qwen2audio_attention_rows.jsonl \
  --out-md outputs/audiojailbreak_origin_smoke20_qwen2audio_attention_summary.md \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --device cuda:0 \
  --layers 0,8,16,24,32
```

Analyze attention deviation:

```bash
python scripts/analyze_attention_deviation.py \
  --input outputs/audiojailbreak_origin_smoke20_qwen2audio_attention_rows.jsonl \
  --out-dir outputs/audiojailbreak_origin_smoke20_qwen2audio_attention_analysis
```

Read:

```bash
cat outputs/audiojailbreak_origin_smoke20_qwen2audio_attention_summary.md
cat outputs/audiojailbreak_origin_smoke20_qwen2audio_attention_analysis/attention_feature_summary.md
```

Main figures:

```text
attention_by_layer_safe_vs_unsafe.png
attention_delta_heatmap.png
attention_deviation_pca.png
```

For stable conclusions, move from `smoke20` to `p100` or `full` after the smoke
run works.

## 9. Common Issues

### All labels are safe

Example:

```text
labels: {0: 20}
```

The router is binary and needs both safe and unsafe classes. Increase the sample
count or run the full benchmark.

### CUDA out of memory

Try sparse hidden extraction:

```bash
--layers 0,8,16,24,32
```

### Generated `.npz` cannot be loaded by system Python

Use the project virtual environment:

```bash
source .venv/bin/activate
python scripts/validate_features.py ...
```

### Hugging Face permission error

Login and confirm gated model access:

```bash
huggingface-cli login
```
