#!/usr/bin/env python3
"""
Template for adapting the hidden-router defense to a new model or benchmark.

This file is intentionally runnable only after you fill in the model-specific
functions marked with TODO. The required output schema is shared by all defense
training scripts in this folder.
"""

import argparse
import json
from pathlib import Path

import numpy as np


POOL_NAMES = np.asarray(["last", "mean_context", "audio_tokens"], dtype=object)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model_and_processor(model_name_or_path, device):
    """
    TODO: Load the target model/tokenizer/processor here.

    Return any object you need for `extract_one`.
    """
    raise NotImplementedError("Fill in model loading for your target model.")


def extract_one(example, model_bundle, device):
    """
    TODO: Run one benchmark example through the target model and return features.

    Required return value:

    {
      "hidden_last": np.ndarray with shape [L, D],
      "hidden_mean_context": np.ndarray with shape [L, D],
      "hidden_audio_tokens": np.ndarray with shape [L, D],
      "scores": np.ndarray with shape [K],
      "score_names": list[str],
      "meta": dict
    }

    Pooling definitions used in this project:

    - hidden_last: hidden state at the final prompt/context token, per layer.
    - hidden_mean_context: mean hidden state over context tokens, per layer.
    - hidden_audio_tokens: mean hidden state over audio tokens when available.

    If the model does not expose audio-token positions, use the best available
    modality span and document that choice in the report.
    """
    raise NotImplementedError("Fill in hidden-state extraction for your target model.")


def label_from_example(example):
    """
    Convert offline judge labels to binary labels.

    Adjust field names if your benchmark uses a different schema.
    """
    label = str(example.get("llamaguard_label") or example.get("judge_label") or "").lower()
    return 1 if label == "unsafe" else 0


def main():
    parser = argparse.ArgumentParser(description="Template extractor for hidden-router defense features.")
    parser.add_argument("--manifest", required=True, help="JSONL manifest with benchmark rows and judge labels")
    parser.add_argument("--model", required=True, help="Target model name or local path")
    parser.add_argument("--out-features", required=True, help="Output hidden_features.npz")
    parser.add_argument("--out-meta", required=True, help="Output hidden_meta.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    bundle = load_model_and_processor(args.model, args.device)

    hidden_last = []
    hidden_mean_context = []
    hidden_audio_tokens = []
    scores = []
    labels = []
    ids = []
    meta_rows = []
    score_names = None

    for i, row in enumerate(rows):
        example_id = str(row.get("id") or i)
        try:
            item = extract_one(row, bundle, args.device)
            if score_names is None:
                score_names = list(item["score_names"])
            elif score_names != list(item["score_names"]):
                raise ValueError("score_names changed across examples")

            hidden_last.append(np.asarray(item["hidden_last"], dtype=np.float16))
            hidden_mean_context.append(np.asarray(item["hidden_mean_context"], dtype=np.float16))
            hidden_audio_tokens.append(np.asarray(item["hidden_audio_tokens"], dtype=np.float16))
            scores.append(np.asarray(item["scores"], dtype=np.float32))
            meta = dict(row)
            meta.update(item.get("meta") or {})
            meta["id"] = example_id
            meta["extract_error"] = None
        except Exception as exc:
            hidden_last.append(None)
            hidden_mean_context.append(None)
            hidden_audio_tokens.append(None)
            scores.append(None)
            meta = dict(row)
            meta["id"] = example_id
            meta["extract_error"] = f"{type(exc).__name__}: {exc}"
        labels.append(label_from_example(row))
        ids.append(example_id)
        meta_rows.append(meta)

    valid_indices = [i for i, value in enumerate(hidden_last) if value is not None]
    valid_count = len(valid_indices)
    if valid_count == 0:
        raise SystemExit("[ERROR] no valid examples were extracted")

    hidden_shape = hidden_last[valid_indices[0]].shape
    score_shape = scores[valid_indices[0]].shape
    for i in range(len(hidden_last)):
        if hidden_last[i] is None:
            hidden_last[i] = np.zeros(hidden_shape, dtype=np.float16)
            hidden_mean_context[i] = np.zeros(hidden_shape, dtype=np.float16)
            hidden_audio_tokens[i] = np.zeros(hidden_shape, dtype=np.float16)
            scores[i] = np.zeros(score_shape, dtype=np.float32)

    out_features = Path(args.out_features)
    out_meta = Path(args.out_meta)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_features,
        hidden_last=np.stack(hidden_last, axis=0),
        hidden_mean_context=np.stack(hidden_mean_context, axis=0),
        hidden_audio_tokens=np.stack(hidden_audio_tokens, axis=0),
        scores=np.stack(scores, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        ids=np.asarray(ids, dtype=object),
        score_names=np.asarray(score_names or [], dtype=object),
        pool_names=POOL_NAMES,
    )

    with open(out_meta, "w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {out_features} with {valid_count} valid examples")
    print(f"[OK] wrote {out_meta} with {len(meta_rows)} metadata rows")


if __name__ == "__main__":
    main()
