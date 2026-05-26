# Hidden-State Router Defense

This folder contains the end-to-end AudioJailbreak hidden-state defense
workflow:

1. build a benchmark manifest
2. run a target audio-language model
3. judge responses with Llama Guard or another safety judge
4. compute the unsafe rate
5. extract target-model hidden states
6. train and simulate the hidden-state router defense

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

For the full from-zero command sequence, start with:

```text
EXPERIMENT_PIPELINE.md
```

## Documentation

| File | Purpose |
|---|---|
| `docs/USAGE.md` | Benchmark-specific run guide for AudioJailbreak, JALMBench, and SACRED-Bench. |
| `docs/ARCHITECTURE.md` | Repository layout, code structure, and artifact naming. |
| `METHOD_DETAILS.md` | Method formulas, metrics, training flow, and defense simulation details. |
| `ADAPTER_GUIDE.md` | How to plug in a new target model or benchmark. |
| `EXPERIMENT_PIPELINE.md` | Full from-zero command sequence. |

## Included Scripts

Benchmark and judging:

| Purpose | Script |
|---|---|
| Build AudioJailbreak manifest | `scripts/build_audiojailbreak_manifest.py` |
| Build generic Hugging Face soundfolder manifest | `scripts/build_hf_soundfolder_manifest.py` |
| Build JALMBench manifest | `scripts/build_jalmbench_manifest.py` |
| Run Qwen2-Audio on AudioJailbreak | `scripts/run_qwen2_audio_audiojailbreak.py` |
| Judge responses with Llama Guard | `scripts/judge_with_llamaguard.py` |
| Summarize unsafe rate | `scripts/summarize_judge_labels.py` |

Hidden-state extraction and defense:

| Purpose | Script |
|---|---|
| Qwen2-Audio hidden extraction | `scripts/extract_qwen2_audio_hidden.py` |
| New target model runner template | `scripts/target_model_runner_template.py` |
| New hidden extractor template | `scripts/extractor_template.py` |
| Validate exported hidden features | `scripts/validate_features.py` |
| Single-layer probe sweep | `scripts/train_hidden_probes.py` |
| Single-router defense simulation | `scripts/simulate_hidden_router_defense.py` |
| Auto-layer router | `scripts/train_auto_layer_router.py` |
| Category-transfer router | `scripts/train_category_transfer_router.py` |

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

For only the router/defense scripts:

```bash
python -m pip install -r requirements.txt
```

For the complete benchmark -> judge -> hidden extraction -> defense workflow:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-experiment.txt
```

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

Optional category-transfer analysis:

```bash
python defense_method/scripts/train_category_transfer_router.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl \
  --out-dir outputs/my_model_category_transfer
```

This trains on one harmful category at a time and evaluates transfer to every
eligible category. It is different from held-out category evaluation, which
trains on all other categories and tests on one held-out category.

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

The judging and defense code are model-agnostic. Response generation and hidden
extraction are model-specific.

For a new model, implement:

```text
scripts/target_model_runner_template.py
scripts/extractor_template.py
```

The runner should produce benchmark rows plus a response field. The extractor
should produce:

```text
hidden_features.npz
hidden_meta.jsonl
```

Included Qwen2-Audio examples:

```text
scripts/run_qwen2_audio_audiojailbreak.py
scripts/extract_qwen2_audio_hidden.py
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
