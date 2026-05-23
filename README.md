# Hidden-State Router Defense

This folder contains the model-agnostic defense code used in the AudioJailbreak
experiments.

The defense learns a lightweight router from a target model's hidden states. The
router predicts whether the target model's response is likely unsafe, then
simulates routing high-risk examples to a safer path.

```text
benchmark audio/text
  -> target model response
  -> offline judge label, for example Llama Guard
  -> target model hidden states
  -> train hidden-state router
  -> simulate defense metrics
```

Llama Guard is only used offline to create training/evaluation labels. At router
time, the defense uses hidden-state features, not Llama Guard.

## What You Need

You only need two input files:

```text
hidden_features.npz
hidden_meta.jsonl
```

`hidden_features.npz` must contain:

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

Conventions:

- `N`: examples.
- `L`: saved model layers.
- `D`: hidden size.
- `labels[i] = 1`: unsafe.
- `labels[i] = 0`: safe.
- `hidden_audio_tokens` can equal `hidden_mean_context` if the model has no
  explicit audio token span.

`hidden_meta.jsonl` must have one JSON object per example. Recommended fields:

```json
{"id": "example-0001", "source": "benchmark", "category": "unknown", "llamaguard_label": "unsafe", "extract_error": null}
```

Rows with non-null `extract_error` are excluded from training/evaluation.

## Install

From the repository root:

```bash
python -m pip install -r defense_method/requirements.txt
```

The core scripts need `numpy`, `scikit-learn`, and `matplotlib`.

## Run

Validate the feature files:

```bash
python defense_method/scripts/validate_features.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl
```

Run the complete defense pipeline:

```bash
FEATURES=path/to/hidden_features.npz \
META=path/to/hidden_meta.jsonl \
OUT_DIR=outputs/my_model_defense \
PYTHON=python \
bash defense_method/run_hidden_router_pipeline.sh
```

Main outputs:

```text
outputs/my_model_defense/single_layer_probe/probe_summary.md
outputs/my_model_defense/single_router_defense/defense_simulation_summary.md
outputs/my_model_defense/auto_layer_router/auto_layer_router_summary.md
```

Read those three markdown files first.

## Output Meaning

- `single_layer_probe`: which layer/pooling choice separates safe vs unsafe best.
- `single_router_defense`: one deployable-style router with a threshold.
- `auto_layer_router`: selects useful layers and ensembles them.

Report these numbers:

- target model and benchmark
- number of valid examples
- offline judge used for labels
- unsafe rate before defense
- best AUROC/AUPRC
- threshold objective, usually `f1`
- unsafe rate after routing
- route rate
- false-route rate on safe examples
- held-out source/category results, if available

## Adapting a New Model

The defense code does not call the target model. For a new model, write an
extractor that produces the two files above.

Start from:

```text
defense_method/scripts/extractor_template.py
```

Project-specific extractors you can copy from:

```text
scripts/extract_audiojailbreak_qwen2_hidden.py
scripts/extract_audiojailbreak_qwen25_omni_hidden.py
scripts/extract_audiojailbreak_ultravox_hidden.py
```

For each benchmark row, the extractor should:

1. Load the same input used during target-model inference.
2. Run the target model with `output_hidden_states=True`.
3. Pool hidden states into `last`, `mean_context`, and `audio_tokens`.
4. Convert the offline judge result to `labels`, where only `unsafe` is `1`.
5. Save `hidden_features.npz` and `hidden_meta.jsonl`.

If you saved only some layers, include real layer ids in the NPZ:

```python
layer_indices=np.array([0, 8, 16, 24, 32], dtype=np.int64)
```

## Useful Options

```bash
FEATURES=path/to/hidden_features.npz \
META=path/to/hidden_meta.jsonl \
OUT_DIR=outputs/my_model_defense \
PYTHON=python \
SEED=42 \
TRAIN_FRAC=0.70 \
OBJECTIVE=f1 \
SPLIT_MODE=all \
PLOT_SWEEP_SPLITS=random \
bash defense_method/run_hidden_router_pipeline.sh
```

`OBJECTIVE`:

- `f1`: balanced default.
- `high_recall`: catches more unsafe examples, usually routes more safe ones.
- `low_route`: available in the auto-layer router.

`SPLIT_MODE`:

- `random`: stratified random split.
- `source`: held-out source splits.
- `category`: held-out category splits.
- `all`: random plus held-out source/category splits.

For formulas and metric definitions, see `METHOD_DETAILS.md`. For adapter
details, see `ADAPTER_GUIDE.md`.
