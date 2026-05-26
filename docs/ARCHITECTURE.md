# File and Code Architecture

This repository contains an end-to-end hidden-state router defense workflow for
audio-language model jailbreak evaluation.

The project is organized around two layers:

```text
defense_method/
├── hidden_router/          # shared library code
├── scripts/                # command-line entrypoints
├── docs/                   # user-facing documentation
├── internal_notes/         # project notes for maintainers
├── outputs/                # generated experiment artifacts, git-ignored
├── run_hidden_router_pipeline.sh
├── requirements.txt
└── requirements-experiment.txt
```

## Top-Level Files

| Path | Purpose |
|---|---|
| `README.md` | Main project overview and quick start. |
| `METHOD_DETAILS.md` | Method definitions, formulas, metrics, and interpretation guardrails. |
| `EXPERIMENT_PIPELINE.md` | Full command sequence for benchmark -> judge -> hidden extraction -> defense. |
| `ADAPTER_GUIDE.md` | How to adapt a new target model or benchmark. |
| `docs/ARCHITECTURE.md` | This file: repository and code structure. |
| `docs/USAGE.md` | User guide with benchmark-specific commands. |
| `run_hidden_router_pipeline.sh` | Runs validation, single-layer probes, single-router simulation, and auto-layer router. |
| `requirements.txt` | Minimal router/defense dependencies. |
| `requirements-experiment.txt` | Extra dependencies for benchmark loading, model inference, judging, and audio processing. |

## Shared Library: `hidden_router/`

`hidden_router/` contains reusable logic shared by multiple scripts. Keeping
these functions in one place prevents metric, split, threshold, and layer logic
from drifting across experiments.

```text
hidden_router/
├── __init__.py
├── io.py
├── layers.py
├── metrics.py
├── splits.py
└── thresholds.py
```

| Module | Responsibility |
|---|---|
| `io.py` | JSONL helpers: `load_jsonl`, `write_jsonl`, `append_jsonl`. |
| `metrics.py` | Shared math: `sigmoid`, `auroc`, `auprc`, `confusion_counts`. |
| `splits.py` | Train/test split utilities: stratified split and held-out source/category splits. |
| `layers.py` | Real model layer id handling, sparse layer mapping, selected layer positions, depth regions. |
| `thresholds.py` | Router threshold metrics, threshold sweeps, threshold selection objectives, route-rate thresholding. |

## Command Entrypoints: `scripts/`

Scripts are the user-facing command-line interface. They should stay thin:
parse arguments, call shared code, and write outputs.

### Benchmark Manifest Builders

| Script | Purpose |
|---|---|
| `scripts/build_audiojailbreak_manifest.py` | Builds a manifest from `MBZUAI/AudioJailbreak`. |
| `scripts/build_jalmbench_manifest.py` | Builds a manifest from JALMBench subsets such as `ADiv` and `SSJ`. |
| `scripts/build_hf_soundfolder_manifest.py` | Builds a manifest from generic Hugging Face soundfolder-style datasets, such as SACRED-Bench. |

### Target Model and Judge

| Script | Purpose |
|---|---|
| `scripts/run_qwen2_audio_audiojailbreak.py` | Runs Qwen2-Audio on any compatible audio manifest and writes responses. |
| `scripts/judge_with_llamaguard.py` | Uses LlamaGuard or a compatible causal LM judge to label responses. |
| `scripts/summarize_judge_labels.py` | Summarizes label counts and unsafe rate. |

### Hidden-State Extraction

| Script | Purpose |
|---|---|
| `scripts/extract_qwen2_audio_hidden.py` | Extracts Qwen2-Audio hidden-state features and metadata. |
| `scripts/extractor_template.py` | Template for implementing a hidden extractor for a new model. |
| `scripts/target_model_runner_template.py` | Template for implementing response generation for a new model. |
| `scripts/validate_features.py` | Validates the hidden feature `.npz` and metadata `.jsonl` schema. |

### Router Training and Defense Simulation

| Script | Purpose |
|---|---|
| `scripts/train_hidden_probes.py` | Sweeps single-layer linear probes over pooling methods and layers. |
| `scripts/simulate_hidden_router_defense.py` | Trains one deployable-style router and simulates route/no-route defense metrics. |
| `scripts/train_auto_layer_router.py` | Selects useful layers automatically and ensembles multiple probes. |
| `scripts/train_category_transfer_router.py` | Trains on one harmful category and evaluates transfer to other categories. |
| `scripts/layer_utils.py` | Compatibility wrapper around `hidden_router.layers`. |

## Generated Artifacts: `outputs/`

`outputs/` is git-ignored. A complete run normally creates:

```text
outputs/{run_name}_manifest.jsonl
outputs/{run_name}_qwen2audio_responses.jsonl
outputs/{run_name}_qwen2audio_llamaguard.jsonl
outputs/{run_name}_qwen2audio_safety_summary.md
outputs/{run_name}_qwen2audio_hidden_features.npz
outputs/{run_name}_qwen2audio_hidden_meta.jsonl
outputs/{run_name}_qwen2audio_defense/
```

The defense directory contains:

```text
single_layer_probe/probe_summary.md
single_router_defense/defense_simulation_summary.md
auto_layer_router/auto_layer_router_summary.md
```

The auto-layer summary is usually the main result to inspect first.

## Data Flow

```text
manifest builder
  -> *_manifest.jsonl

target model runner
  -> *_responses.jsonl

safety judge
  -> *_llamaguard.jsonl
  -> *_safety_summary.md

hidden extractor
  -> *_hidden_features.npz
  -> *_hidden_meta.jsonl

router pipeline
  -> *_defense/single_layer_probe/
  -> *_defense/single_router_defense/
  -> *_defense/auto_layer_router/
```

## Feature Contract

All router scripts consume two files:

```text
hidden_features.npz
hidden_meta.jsonl
```

The `.npz` file must contain:

```text
hidden_last          [N, L, D]
hidden_mean_context  [N, L, D]
hidden_audio_tokens  [N, L, D]
scores               [N, K]
labels               [N]
ids                  [N]
score_names          [K]
pool_names           [3]
```

`labels[i] = 1` means unsafe, and `labels[i] = 0` means safe. Rows with
non-null `extract_error` in metadata are excluded from training and evaluation.

## Design Rules

- Keep command-line scripts as entrypoints; put reusable logic in `hidden_router/`.
- Keep benchmark-specific code in manifest builders.
- Keep model-specific response generation and hidden extraction separate from router training.
- Keep metrics, splits, layer mapping, and threshold logic shared.
- Do not commit generated `outputs/` artifacts.

