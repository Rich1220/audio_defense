# Router + Interpretability + Future LoRA Plan

This plan defines the next-stage research direction for the hidden-state router
defense. The immediate goal is Phase 1: build a stronger internal-signal
analysis pipeline on AudioJailbreak. LoRA is kept as a later phase because we
need a balanced harmful/benign supervised training set before safety fine-tuning
is meaningful.

## Motivation

The current defense already trains a hidden-state router:

```text
audio prompt
  -> target audio-language model
  -> hidden states
  -> lightweight router risk score
  -> simulated route / no-route defense
```

This is useful, but we need more evidence for why the router works and what
failure mode it detects. The next step is to connect router behavior with
interpretable internal signals:

```text
layer-wise linear probes
PCA / representation clusters
attention-to-audio vs attention-to-context deviation
case-level visualizations
```

The long-term goal is:

```text
router detects risk
  -> high-risk examples activate a safer path
  -> future option: router-gated safety LoRA
```

## Why Not LoRA Immediately

LoRA needs a balanced supervised dataset. If we fine-tune only on harmful audio
with refusal targets, the adapter may learn broad over-refusal:

```text
harmful audio -> refuse
benign audio  -> should remain helpful
```

We currently have harmful/jailbreak benchmark labels and router features, but
we do not yet have a carefully prepared benign audio instruction set with
helpful target responses. Therefore LoRA should wait until Phase 2.

## Phase 1: Router + Interpretability

Phase 1 uses AudioJailbreak and the current Qwen2-Audio pipeline.

### Goals

1. Train and evaluate the hidden-state router on AudioJailbreak.
2. Produce layer-wise linear probe figures.
3. Produce hidden-state PCA plots.
4. Extract attention mass over audio/context spans.
5. Measure attention deviation:

```text
attention_deviation = attention_to_audio - attention_to_context
```

6. Compare safe vs unsafe examples using:

```text
layer-wise attention curves
unsafe - safe attention delta heatmaps
attention-deviation PCA
case-level summaries
```

### Expected Artifacts

Router artifacts:

```text
outputs/audiojailbreak_origin_{size}_qwen2audio_defense/
  single_layer_probe/probe_summary.md
  single_layer_probe/layerwise_auroc_by_pooling.png
  single_layer_probe/best_hidden_pca_safe_vs_unsafe.png
  auto_layer_router/auto_layer_router_summary.md
```

Attention artifacts:

```text
outputs/audiojailbreak_origin_{size}_qwen2audio_attention_rows.jsonl
outputs/audiojailbreak_origin_{size}_qwen2audio_attention_summary.md
outputs/audiojailbreak_origin_{size}_qwen2audio_attention_analysis/
  attention_by_layer_safe_vs_unsafe.png
  attention_delta_heatmap.png
  attention_deviation_pca.png
  attention_feature_summary.md
```

### Main Research Questions

1. Which layers and pooling choices best separate safe vs unsafe outputs?
2. Do unsafe examples attend more to audio tokens than safe examples?
3. Does the unsafe cluster in hidden-state PCA align with attention deviation?
4. Are there categories/source attacks where router risk and attention deviation disagree?
5. Are failures caused by audio over-attention, context under-attention, or later decoding-mode shifts?

## Phase 2: Build Balanced LoRA Data

LoRA requires a supervised training set with both harmful and benign audio.

### Harmful Side

Candidate sources:

```text
AudioJailbreak unsafe examples
JALMBench unsafe examples
SACRED-Bench unsafe examples
```

Target response:

```text
short safe refusal
```

### Benign Side

Candidate sources:

```text
benign audio QA
harmless instruction TTS
AudioCaps / Clotho-style benign audio descriptions
internally generated harmless spoken requests
```

Target response:

```text
helpful answer
```

### Required Splits

We need strict splits:

```text
LoRA train
LoRA validation
router validation
final held-out test
```

The final router evaluation set should not be used for LoRA training.

## Phase 3: Router-Gated LoRA

Once Phase 2 data is ready, compare:

```text
base model
always-on safety LoRA
router-gated safety LoRA
```

Deployment-style policy:

```text
if router_score >= threshold:
    activate safety LoRA or route to safety path
else:
    use base model
```

Metrics:

```text
unsafe rate
unsafe reduction
route rate
safe false-route rate
benign over-refusal
helpfulness retention
```

## Phase 4: Joint Explanation

After LoRA is available, compare representations before and after adapter use:

```text
base hidden states
LoRA hidden states
attention deviation before/after LoRA
router scores before/after LoRA
```

Useful figures:

```text
base vs LoRA PCA
unsafe cluster shift toward safe/refusal cluster
attention audio-context delta before/after LoRA
router score distribution before/after LoRA
```

## Immediate Phase 1 Command Skeleton

Use AudioJailbreak smoke first:

```bash
python scripts/build_audiojailbreak_manifest.py \
  --config Origin \
  --split origin \
  --limit 20 \
  --download-audio \
  --audio-dir outputs/audiojailbreak_audio \
  --out outputs/audiojailbreak_origin_smoke20_manifest.jsonl
```

Then run:

```text
Qwen2-Audio response
LlamaGuard judge
hidden extraction
run_hidden_router_pipeline.sh
attention extraction
attention analysis
```

For meaningful plots, use a larger sample once the smoke run works:

```text
limit 100 or full run
```

Smoke runs can have too few unsafe examples for stable interpretation.

