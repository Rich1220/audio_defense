#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hidden_router.io import load_jsonl
from hidden_router.layers import layer_indices_for


REQUIRED_ARRAYS = [
    "hidden_last",
    "hidden_mean_context",
    "hidden_audio_tokens",
    "scores",
    "labels",
    "ids",
    "score_names",
    "pool_names",
]


def fail(message):
    raise SystemExit(f"[ERROR] {message}")


def main():
    parser = argparse.ArgumentParser(description="Validate hidden-router defense feature files.")
    parser.add_argument("--features", required=True, help="Path to hidden_features.npz")
    parser.add_argument("--meta", required=True, help="Path to hidden_meta.jsonl")
    args = parser.parse_args()

    features_path = Path(args.features)
    meta_path = Path(args.meta)
    if not features_path.exists():
        fail(f"features file not found: {features_path}")
    if not meta_path.exists():
        fail(f"meta file not found: {meta_path}")

    data = np.load(features_path, allow_pickle=True)
    missing = [name for name in REQUIRED_ARRAYS if name not in data.files]
    if missing:
        fail(f"missing arrays in npz: {missing}")

    hidden_last = data["hidden_last"]
    hidden_mean_context = data["hidden_mean_context"]
    hidden_audio_tokens = data["hidden_audio_tokens"]
    scores = data["scores"]
    labels = data["labels"]
    ids = data["ids"]
    score_names = data["score_names"]
    pool_names = [str(x) for x in data["pool_names"].tolist()]
    meta = load_jsonl(meta_path)

    if hidden_last.ndim != 3:
        fail(f"hidden_last must be [N, L, D], got {hidden_last.shape}")
    if hidden_mean_context.shape != hidden_last.shape:
        fail(f"hidden_mean_context shape {hidden_mean_context.shape} != hidden_last {hidden_last.shape}")
    if hidden_audio_tokens.shape != hidden_last.shape:
        fail(f"hidden_audio_tokens shape {hidden_audio_tokens.shape} != hidden_last {hidden_last.shape}")
    if scores.ndim != 2:
        fail(f"scores must be [N, K], got {scores.shape}")

    n = hidden_last.shape[0]
    checks = {
        "scores": scores.shape[0],
        "labels": labels.shape[0],
        "ids": ids.shape[0],
        "meta rows": len(meta),
    }
    bad = {name: count for name, count in checks.items() if count != n}
    if bad:
        fail(f"row-count mismatch against hidden N={n}: {bad}")
    if len(score_names) != scores.shape[1]:
        fail(f"score_names length {len(score_names)} != scores K={scores.shape[1]}")

    expected_pools = {"last", "mean_context", "audio_tokens"}
    if set(pool_names) != expected_pools:
        fail(f"pool_names must be {sorted(expected_pools)}, got {pool_names}")

    unique_labels = sorted(set(int(x) for x in labels.tolist()))
    if not set(unique_labels).issubset({0, 1}):
        fail(f"labels must be binary 0/1, got {unique_labels}")

    extract_errors = sum(1 for row in meta if row.get("extract_error"))
    label_counts = Counter(int(x) for x in labels.tolist())

    layer_ids = layer_indices_for(data, "hidden_last")

    print("[OK] feature schema is valid")
    print(f"[INFO] rows: {n}")
    print(f"[INFO] hidden shape: layers={hidden_last.shape[1]} dim={hidden_last.shape[2]}")
    print(f"[INFO] layer ids: {layer_ids}")
    print(f"[INFO] scores: {scores.shape[1]} columns = {[str(x) for x in score_names.tolist()]}")
    print(f"[INFO] labels: {dict(label_counts)}")
    print(f"[INFO] meta extract_error rows: {extract_errors}")


if __name__ == "__main__":
    main()
