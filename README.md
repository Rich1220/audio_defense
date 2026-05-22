# Hidden-State Router Defense Toolkit

This folder is a reusable toolkit for the hidden-state router defense used in
the AudioJailbreak experiments.

It includes both documentation and runnable code, so another model or benchmark
can use the same method after exporting hidden-state features in the expected
schema.

## Folder Contents

```text
defense_method/
  README.md
  METHOD_DETAILS.md
  ADAPTER_GUIDE.md
  requirements.txt
  run_hidden_router_pipeline.sh
  scripts/
    validate_features.py
    train_hidden_probes.py
    simulate_hidden_router_defense.py
    train_auto_layer_router.py
    extractor_template.py
```

Core scripts:

| Purpose | Script |
|---|---|
| Validate exported features | `scripts/validate_features.py` |
| Single-layer linear probe sweep | `scripts/train_hidden_probes.py` |
| Single-layer router defense simulation | `scripts/simulate_hidden_router_defense.py` |
| Model-specific multi-layer auto-router | `scripts/train_auto_layer_router.py` |
| New model extractor starting point | `scripts/extractor_template.py` |

## Method Summary

The method trains a lightweight linear router that predicts whether a model
response is likely unsafe from the target model's own hidden states.

```text
benchmark input
  -> target model inference
  -> offline safety judge labels responses
  -> extract hidden-state features
  -> train linear router
  -> simulate defense:
       if router_score >= threshold:
           route to refusal / safer model / moderation path
       else:
           keep original response
```

Important: LlamaGuard or another safety judge is used only offline to create
labels. At deployment time, the router uses hidden features, not judge labels.


## GitHub Quickstart for New Users

A new user should think of this toolkit as two stages:

```text
Stage 1: export hidden features from their own model
Stage 2: run the defense scripts on those exported features
```

The defense scripts do not call the target model directly. They only consume two files:

```text
hidden_features.npz
hidden_meta.jsonl
```

### Step 1: Export Features

The user writes or adapts an extractor for their model. A starter template is here:

```text
defense_method/scripts/extractor_template.py
```

The extractor must save these arrays in `hidden_features.npz`:

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

`labels` are offline safety labels:

```text
1 = unsafe
0 = safe
```

The metadata file `hidden_meta.jsonl` must contain one JSON object per row. At minimum:

```json
{"id": "example-1", "source": "benchmark", "category": "harm category", "extract_error": null}
```

If the extractor only saves a subset of layers, include the real model layer ids:

```python
layer_indices=np.array([0, 8, 16, 24, 32], dtype=np.int64)
```

Without `layer_indices`, the scripts assume the layer axis means `0..L-1`.

### Step 2: Validate Features

```bash
python defense_method/scripts/validate_features.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl
```

If this passes, the user can run the full pipeline.

### Step 3: Run the Full Defense Pipeline

```bash
FEATURES=path/to/hidden_features.npz \
META=path/to/hidden_meta.jsonl \
OUT_DIR=outputs/my_model_defense \
PYTHON=python \
bash defense_method/run_hidden_router_pipeline.sh
```

This produces three result folders:

```text
OUT_DIR/single_layer_probe/
OUT_DIR/single_router_defense/
OUT_DIR/auto_layer_router/
```

### Step 4: Read Results

Start with these files:

```text
OUT_DIR/single_layer_probe/probe_summary.md
OUT_DIR/single_router_defense/defense_simulation_summary.md
OUT_DIR/auto_layer_router/auto_layer_router_summary.md
```

Interpretation:

- `single_layer_probe`: which one layer best separates safe vs unsafe.
- `single_router_defense`: deployable-style simulation using the best single-layer router.
- `auto_layer_router`: automatically selects useful layers from available hidden states and ensembles them.

### Step 5: Choose What to Report

For a paper or experiment log, report:

- target model and benchmark
- number of examples and unsafe rate before defense
- which safety judge created labels
- best single-layer AUROC/AUPRC
- unsafe before/after defense
- route rate and false-route rate
- held-out source/category results
- whether features included all layers or sparse layers

## Install

From the repository root:

```bash
cd /mnt/disk2/rich1220/projects/audio_emotion_jailbreak

.venv_qwen2audio/bin/python -m pip install -r defense_method/requirements.txt
```

If your environment already has `numpy`, `scikit-learn`, and `matplotlib`, this
step may already be satisfied.

## Required Input Schema

Prepare two files:

```text
hidden_features.npz
hidden_meta.jsonl
```

`hidden_features.npz` must contain:

```text
hidden_last          shape: [N, L, D]
hidden_mean_context  shape: [N, L, D]
hidden_audio_tokens  shape: [N, L, D]
scores               shape: [N, K]
labels               shape: [N]
ids                  shape: [N]
score_names          shape: [K]
pool_names           shape: [3]
```

`labels[i] = 1` means unsafe. `labels[i] = 0` means safe.

`hidden_meta.jsonl` must have one JSON object per example. Recommended fields:

```json
{
  "id": "unique-example-id",
  "category": "harmful category",
  "source": "benchmark source or attack family",
  "attack_type": "manual / black_box / transfer / etc.",
  "prompt": "original prompt if available",
  "local_audio": "path/to/audio.wav",
  "llamaguard_label": "unsafe",
  "extract_error": null
}
```

Rows with non-null `extract_error` are automatically excluded by the training
scripts.

## One-Command Pipeline

Run the whole defense analysis:

```bash
cd /mnt/disk2/rich1220/projects/audio_emotion_jailbreak

FEATURES=outputs/final_analysis/audiojailbreak_origin_full_ultravox_hidden_features.npz \
META=outputs/final_analysis/audiojailbreak_origin_full_ultravox_hidden_meta.jsonl \
OUT_DIR=outputs/final_analysis/my_hidden_router_defense \
bash defense_method/run_hidden_router_pipeline.sh
```

Optional settings:

```bash
FEATURES=path/to/hidden_features.npz \
META=path/to/hidden_meta.jsonl \
OUT_DIR=outputs/final_analysis/my_model_defense \
PYTHON=.venv_qwen2audio/bin/python \
SEED=42 \
TRAIN_FRAC=0.70 \
OBJECTIVE=f1 \
SPLIT_MODE=all \
PLOT_SWEEP_SPLITS=random,source=PAIR,source=jailbreak_llms \
bash defense_method/run_hidden_router_pipeline.sh
```

`OBJECTIVE` choices:

- `f1`: balanced precision/recall threshold.
- `high_recall`: catches more unsafe examples, usually routes more safe ones.
- `low_route`: available for auto-layer router, prioritizes lower route rate.

`SPLIT_MODE` choices for auto-layer:

- `random`: random stratified split only.
- `source`: held-out source splits.
- `category`: held-out category splits.
- `all`: random plus held-out source/category splits.

## Run Scripts Separately

Validate schema:

```bash
.venv_qwen2audio/bin/python defense_method/scripts/validate_features.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl
```

Single-layer probe sweep:

```bash
.venv_qwen2audio/bin/python defense_method/scripts/train_hidden_probes.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl \
  --out-dir outputs/final_analysis/my_model_single_layer_probe \
  --seed 42
```

Single-layer defense simulation:

```bash
.venv_qwen2audio/bin/python defense_method/scripts/simulate_hidden_router_defense.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl \
  --out-dir outputs/final_analysis/my_model_single_router_defense \
  --seed 42 \
  --objective f1
```

Model-specific auto-layer router:

```bash
.venv_qwen2audio/bin/python defense_method/scripts/train_auto_layer_router.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl \
  --out-dir outputs/final_analysis/my_model_auto_layer_router \
  --split-mode all \
  --selection-mode depth_regions \
  --metric auroc \
  --objective f1 \
  --seed 42 \
  --plot-sweep-splits random
```

## Adapting to Another Model

For a new target model, implement feature extraction first.

Start from:

```text
defense_method/scripts/extractor_template.py
```

The extractor's job is only to produce the schema above. The training and
evaluation scripts are model-agnostic after that.

Existing project-specific extractors that can be used as references:

```text
scripts/extract_audiojailbreak_qwen2_hidden.py
scripts/extract_audiojailbreak_qwen25_omni_hidden.py
scripts/extract_audiojailbreak_ultravox_hidden.py
```

## Output Files

Single-layer probe:

```text
single_layer_probe/probe_summary.md
single_layer_probe/probe_results.json
single_layer_probe/layerwise_auroc_by_pooling.png
single_layer_probe/best_probe_threshold_tradeoff.png
single_layer_probe/best_hidden_pca_safe_vs_unsafe.png
```

Single-router defense:

```text
single_router_defense/defense_simulation_summary.md
single_router_defense/defense_simulation.json
single_router_defense/defense_before_after_unsafe_rate.png
single_router_defense/defense_route_tradeoff.png
```

Auto-layer router:

```text
auto_layer_router/auto_layer_router_summary.md
auto_layer_router/auto_layer_router_results.json
auto_layer_router/auto_layer_selected_layers.csv
auto_layer_router/auto_layer_ensemble_before_after.png
auto_layer_router/auto_layer_route_tradeoff.png
auto_layer_router/auto_layer_auroc_by_split.png
```

## What to Report

At minimum, report:

- Benchmark name and row count.
- Target model and inference prompt format.
- Safety judge used to create labels.
- Overall unsafe rate before defense.
- Best random-split single-layer AUROC/AUPRC.
- Router threshold objective.
- Unsafe rate before/after routing.
- Route rate.
- False-route rate on safe examples.
- Held-out source/category results when available.
- Whether results are offline simulation or integrated live routing.

For full formulas and metric definitions, read `METHOD_DETAILS.md`.
