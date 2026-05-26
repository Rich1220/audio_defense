#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPANS = [
    ("audio", "Audio"),
    ("context", "Context"),
    ("safety_prompt", "Safety prompt"),
    ("assistant_boundary", "Assistant boundary"),
    ("audio_minus_context", "Audio - Context"),
    ("audio_minus_safety_prompt", "Audio - Safety"),
]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    return float(np.mean(values)) if values else float("nan")


def pca2(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def collect_layer_values(rows):
    values = defaultdict(list)
    valid_rows = [row for row in rows if not row.get("extract_error")]
    for row in valid_rows:
        label = str(row.get("llamaguard_label") or "missing")
        for layer in row.get("layer_attention", []):
            layer_id = int(layer["layer"])
            values[(label, layer_id, "audio")].append(float(layer.get("attn_audio", 0.0)))
            values[(label, layer_id, "context")].append(float(layer.get("attn_context", 0.0)))
            values[(label, layer_id, "safety_prompt")].append(float(layer.get("attn_safety_prompt", 0.0)))
            values[(label, layer_id, "assistant_boundary")].append(float(layer.get("attn_assistant_boundary", 0.0)))
            values[(label, layer_id, "audio_minus_context")].append(float(layer.get("attn_audio_minus_context", 0.0)))
            values[(label, layer_id, "audio_minus_safety_prompt")].append(float(layer.get("attn_audio_minus_safety_prompt", 0.0)))
    layers = sorted({key[1] for key in values})
    labels = sorted({key[0] for key in values})
    return valid_rows, values, layers, labels


def plot_by_layer(values, layers, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True)
    axes = axes.reshape(-1)
    for ax, (span, title) in zip(axes, SPANS):
        for label, color in [("safe", "#2f5d9b"), ("unsafe", "#b23b3b")]:
            series = [mean(values.get((label, layer, span), [])) for layer in layers]
            ax.plot(layers, series, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Attention mass")
    axes[3].set_ylabel("Attention mass")
    for ax in axes[3:]:
        ax.set_xlabel("Layer")
    axes[0].legend(frameon=False)
    fig.suptitle("Attention Span Features by Layer")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(values, layers, out_path):
    rows = []
    names = []
    for span, title in SPANS:
        safe = np.array([mean(values.get(("safe", layer, span), [])) for layer in layers])
        unsafe = np.array([mean(values.get(("unsafe", layer, span), [])) for layer in layers])
        rows.append(unsafe - safe)
        names.append(title)
    data = np.asarray(rows, dtype=float)
    finite = data[np.isfinite(data)]
    vmax = max(abs(float(np.min(finite))), abs(float(np.max(finite)))) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Layer")
    ax.set_title("Unsafe - Safe Attention Feature Delta")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def row_feature_vector(row):
    features = []
    for layer in row.get("layer_attention", []):
        for key in [
            "attn_audio",
            "attn_context",
            "attn_safety_prompt",
            "attn_assistant_boundary",
            "attn_audio_minus_context",
            "attn_audio_minus_safety_prompt",
        ]:
            features.append(float(layer.get(key, 0.0)))
    return features


def plot_pca(rows, out_path):
    valid = [row for row in rows if row.get("layer_attention")]
    if len(valid) < 3:
        return False
    lengths = {len(row_feature_vector(row)) for row in valid}
    if len(lengths) != 1:
        return False
    x = np.asarray([row_feature_vector(row) for row in valid], dtype=np.float64)
    coords = pca2(x)
    colors = {"safe": "#2f5d9b", "unsafe": "#b23b3b"}
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    for label in sorted({str(row.get("llamaguard_label")) for row in valid}):
        idx = [i for i, row in enumerate(valid) if str(row.get("llamaguard_label")) == label]
        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=28,
            alpha=0.8,
            label=label,
            color=colors.get(label, "#555555"),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of Attention-Deviation Features")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def write_summary(path, rows, values, layers, labels, pca_written):
    lines = ["# Attention-Deviation Analysis\n"]
    lines.append(f"- valid rows: `{len(rows)}`")
    lines.append(f"- labels: `{labels}`")
    lines.append(f"- layers: `{layers}`")
    lines.append("")
    lines.append("## Mean Features by Label\n")
    lines.append("| Label | Count | Audio | Context | Safety Prompt | Audio-Context | Audio-Safety |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        label_rows = [row for row in rows if str(row.get("llamaguard_label")) == label]
        lines.append(
            f"| {label} | {len(label_rows)} | "
            f"{mean([row.get('mean_attn_audio', 0.0) for row in label_rows]):.4f} | "
            f"{mean([row.get('mean_attn_context', 0.0) for row in label_rows]):.4f} | "
            f"{mean([row.get('mean_attn_safety_prompt', 0.0) for row in label_rows]):.4f} | "
            f"{mean([row.get('mean_attn_audio_minus_context', 0.0) for row in label_rows]):.4f} | "
            f"{mean([row.get('mean_attn_audio_minus_safety_prompt', 0.0) for row in label_rows]):.4f} |"
        )
    lines.append("")
    lines.append("## Figures\n")
    lines.append("- `attention_by_layer_safe_vs_unsafe.png`")
    lines.append("- `attention_delta_heatmap.png`")
    if pca_written:
        lines.append("- `attention_deviation_pca.png`")
    lines.append("")
    lines.append("Interpretation note: `Audio - Context` greater than zero means the last context token attends more to audio-token positions than to the tracked context spans. If unsafe examples do not show higher audio attention, the router may be detecting a later response-mode or refusal-mode shift rather than simple audio over-attention.")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze safe/unsafe attention deviation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.input)
    valid_rows, values, layers, labels = collect_layer_values(rows)
    if not layers:
        raise SystemExit("[ERROR] no layer_attention found")
    plot_by_layer(values, layers, out_dir / "attention_by_layer_safe_vs_unsafe.png")
    plot_delta_heatmap(values, layers, out_dir / "attention_delta_heatmap.png")
    pca_written = plot_pca(valid_rows, out_dir / "attention_deviation_pca.png")
    write_summary(out_dir / "attention_feature_summary.md", valid_rows, values, layers, labels, pca_written)
    print("[OK] wrote:", out_dir)


if __name__ == "__main__":
    main()

