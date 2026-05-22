# Adapter Guide: Using the Defense Method With New Models or Benchmarks

This guide explains how to plug in a new target model or benchmark.

## 1. The Only Required Contract

The training scripts do not need to know your model architecture.

They only require that your extractor writes:

```text
hidden_features.npz
hidden_meta.jsonl
```

with the schema described in `README.md` and `METHOD_DETAILS.md`.

Therefore, adapting a new model means writing a feature extractor.

## 2. Model Adapter Responsibilities

For each benchmark row, your adapter should:

1. Load the input.
2. Build the same prompt/context used during target-model inference.
3. Run the model with hidden states enabled.
4. Identify valid context tokens.
5. Identify audio-token positions if available.
6. Pool hidden states into:
   - `hidden_last`
   - `hidden_mean_context`
   - `hidden_audio_tokens`
7. Optionally compute prefix/logit scores.
8. Save metadata and extraction errors.

## 3. Minimal Pseudocode

```python
rows = load_jsonl(judged_response_file)

hidden_last = []
hidden_mean_context = []
hidden_audio_tokens = []
scores = []
meta_rows = []

for row in rows:
    try:
        inputs = build_model_inputs(row)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        hidden_states = outputs.hidden_states
        attention_mask = inputs["attention_mask"][0]

        last = pool_last(hidden_states, attention_mask)
        mean_context = pool_mean_context(hidden_states, attention_mask)
        audio_tokens = pool_audio_tokens(hidden_states, inputs)

        hidden_last.append(last)
        hidden_mean_context.append(mean_context)
        hidden_audio_tokens.append(audio_tokens)
        scores.append(compute_optional_prefix_scores(outputs.logits, tokenizer))

        row["extract_error"] = None

    except Exception as exc:
        hidden_last.append(zero_hidden())
        hidden_mean_context.append(zero_hidden())
        hidden_audio_tokens.append(zero_hidden())
        scores.append([0.0, 0.0, 0.0])
        row["extract_error"] = f"{type(exc).__name__}: {exc}"

    meta_rows.append(row)

np.savez_compressed(
    out_npz,
    hidden_last=np.stack(hidden_last).astype(np.float16),
    hidden_mean_context=np.stack(hidden_mean_context).astype(np.float16),
    hidden_audio_tokens=np.stack(hidden_audio_tokens).astype(np.float16),
    scores=np.asarray(scores, dtype=np.float32),
    labels=np.asarray([1 if r["llamaguard_label"] == "unsafe" else 0 for r in meta_rows]),
    ids=np.asarray([r["id"] for r in meta_rows], dtype=object),
    score_names=np.asarray(["refusal", "transcription", "answer"], dtype=object),
    pool_names=np.asarray(["last", "mean_context", "audio_tokens"], dtype=object),
)
write_jsonl(out_meta, meta_rows)
```

## 4. How to Build Labels

The current convention assumes:

```text
llamaguard_label = "unsafe" -> label 1
everything else             -> label 0
```

If using another judge, normalize labels to the same convention:

```text
safe
unsafe
judge_error
empty_response
```

Then map only `unsafe` to 1.

## 5. If Your Model Is Text-Only

The method still works.

Set:

```text
hidden_audio_tokens = hidden_mean_context
```

or create another meaningful token subset, such as user-prompt tokens.

Document this in your report.

## 6. If Your Model Does Not Expose Audio Tokens

Use one of these options:

1. If audio embeddings are inserted into the language-model context, track the insertion span.
2. If audio is encoded separately and fused later, pool the fused token positions closest to audio.
3. If no token span is available, set `hidden_audio_tokens = hidden_mean_context`.

The scripts will run as long as the shape is `[N, L, D]`.

## 7. If Your Model Has Different Layer Counts or Sparse Layers

That is fine. The scripts read the available layer axis from:

```text
hidden_last.shape[1]
```

If you store every layer, no extra metadata is required. The scripts will treat the layer ids as `0..L-1`.

If you store only a subset of layers, add the real model layer ids to the npz file. For example, if the hidden arrays contain only model layers 0, 8, 16, 24, and 32:

```python
np.savez_compressed(
    out_path,
    hidden_last=hidden_last,
    hidden_mean_context=hidden_mean_context,
    hidden_audio_tokens=hidden_audio_tokens,
    layer_indices=np.array([0, 8, 16, 24, 32], dtype=np.int64),
    scores=scores,
    labels=labels,
    ids=ids,
    score_names=score_names,
    pool_names=pool_names,
)
```

Supported mapping keys are:

```text
layer_indices
hidden_layer_indices
layers
hidden_last_layer_indices
hidden_mean_context_layer_indices
hidden_audio_tokens_layer_indices
```

Single-layer probes, hidden-router defense, and auto-layer/multiple-layer defense will train only on the available layers and will report the real layer ids. Auto-layer depth regions are computed over the available layer positions, so sparse features still work.

Auto-layer selection is recommended because layer indices do not transfer across models.

## 8. If Your Benchmark Is Not AudioJailbreak

You can still use the method.

Recommended metadata fields:

```json
{
  "id": "benchmark/example/0001",
  "benchmark": "my_benchmark",
  "source": "attack_family_or_dataset_split",
  "category": "harmful_topic",
  "attack_type": "manual",
  "prompt": "input prompt text if available",
  "local_audio": "path/to/audio.wav if available",
  "model_response": "...",
  "llamaguard_label": "safe",
  "extract_error": null
}
```

If fields are missing:

- Set `source` to the benchmark name.
- Set `category` to `"unknown"`.
- Set `attack_type` to `"unknown"`.

Random split does not require source/category fields. Held-out evaluation does.

## 9. Recommended Output Directory Naming

Use a model and benchmark specific prefix:

```text
outputs/final_analysis/{benchmark}_{model}_hidden_features.npz
outputs/final_analysis/{benchmark}_{model}_hidden_meta.jsonl
outputs/final_analysis/{benchmark}_{model}_hidden_probe/
outputs/final_analysis/{benchmark}_{model}_hidden_router_defense/
outputs/final_analysis/{benchmark}_{model}_auto_layer_router/
```

Example:

```text
outputs/final_analysis/audiojailbreak_origin_full_ultravox_hidden_features.npz
outputs/final_analysis/audiojailbreak_origin_full_ultravox_hidden_meta.jsonl
outputs/final_analysis/audiojailbreak_origin_full_ultravox_hidden_probe/
```

## 10. Validation Checklist

Before training:

```bash
.venv_qwen2audio/bin/python defense_method/scripts/validate_features.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl
```

Expected:

- All hidden arrays have the same `[N, L, D]` shape.
- `len(meta_rows) == N`.
- Labels are 0/1.
- Extract errors are small or explained.

## 11. Recommended First Run

Start with random split only:

```bash
.venv_qwen2audio/bin/python defense_method/scripts/train_auto_layer_router.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl \
  --out-dir outputs/final_analysis/test_auto_layer_router_random \
  --split-mode random \
  --selection-mode depth_regions \
  --metric auroc \
  --objective f1 \
  --seed 42 \
  --plot-sweep-splits random
```

Then run all splits:

```bash
.venv_qwen2audio/bin/python defense_method/scripts/train_auto_layer_router.py \
  --features path/to/hidden_features.npz \
  --meta path/to/hidden_meta.jsonl \
  --out-dir outputs/final_analysis/test_auto_layer_router_all \
  --split-mode all \
  --selection-mode depth_regions \
  --metric auroc \
  --objective f1 \
  --seed 42 \
  --plot-sweep-splits random
```

## 12. Reporting Checklist

For each new model/benchmark, report:

- Target model.
- Benchmark.
- Number of rows.
- Judge model and label mapping.
- Overall unsafe rate.
- Feature extraction errors.
- Best single-layer pooling/layer.
- Best single-layer AUROC/AUPRC.
- Random split defense reduction.
- Route rate.
- Safe false-route rate.
- Source-heldout results if source exists.
- Category-heldout results if category exists.
- Auto-layer selected layers.
- Limitations and human-audit caveats.
