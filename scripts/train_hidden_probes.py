#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from layer_utils import layer_positions, selected_position_layers


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def auroc(y, scores):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auprc(y, scores):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def threshold_metrics(y, scores, thresholds):
    rows = []
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": threshold,
                "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "route_rate": float(pred.mean()) if len(pred) else 0.0,
                "residual_unsafe_rate": fn / len(y) if len(y) else 0.0,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return rows


def fit_predict(x_train, y_train, x_test, seed):
    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    x_test_z = scaler.transform(x_test)
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=300,
        random_state=seed,
    )
    clf.fit(x_train_z, y_train.astype(int))
    return clf.predict_proba(x_test_z)[:, 1], {
        "mean": scaler.mean_,
        "std": scaler.scale_,
        "coef": clf.coef_[0],
        "bias": float(clf.intercept_[0]),
    }


def stratified_random_split(y, train_frac, seed):
    rng = random.Random(seed)
    pos = [i for i, v in enumerate(y) if int(v) == 1]
    neg = [i for i, v in enumerate(y) if int(v) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    train = pos[: int(round(len(pos) * train_frac))] + neg[: int(round(len(neg) * train_frac))]
    test = pos[int(round(len(pos) * train_frac)) :] + neg[int(round(len(neg) * train_frac)) :]
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def heldout_splits(meta, key, y, min_test_pos=2, min_train_pos=2):
    groups = defaultdict(list)
    for i, row in enumerate(meta):
        groups[str(row.get(key) or "None")].append(i)
    splits = []
    for value, test_idx in sorted(groups.items()):
        test_idx = np.asarray(test_idx, dtype=int)
        train_idx = np.asarray([i for i in range(len(meta)) if i not in set(test_idx.tolist())], dtype=int)
        if y[test_idx].sum() < min_test_pos or y[train_idx].sum() < min_train_pos:
            continue
        if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
            continue
        splits.append((f"{key}={value}", train_idx, test_idx))
    return splits


def pca2(x):
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", path)


def plot_layerwise(results, out_dir):
    split = "random"
    rows = [r for r in results if r["split"] == split and r["feature_set"] == "hidden"]
    pools = sorted({r["pooling"] for r in rows})
    plt.figure(figsize=(9, 5))
    for pool in pools:
        rs = sorted([r for r in rows if r["pooling"] == pool], key=lambda r: r["layer"])
        plt.plot([r["layer"] for r in rs], [r["auroc"] for r in rs], marker="o", markersize=3, label=pool)
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.ylim(0.0, 1.0)
    plt.title("Layer-wise Linear Probe AUROC")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    savefig(out_dir / "layerwise_auroc_by_pooling.png")


def plot_split_heatmap(results, out_dir):
    rows = [r for r in results if r["feature_set"] == "hidden"]
    split_names = sorted({r["split"] for r in rows})
    pools = sorted({r["pooling"] for r in rows})
    data = np.zeros((len(split_names), len(pools)))
    for i, split in enumerate(split_names):
        for j, pool in enumerate(pools):
            vals = [r["auroc"] for r in rows if r["split"] == split and r["pooling"] == pool]
            data[i, j] = np.nanmax(vals) if vals else np.nan
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(split_names))))
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(pools)))
    ax.set_xticklabels(pools)
    ax.set_yticks(np.arange(len(split_names)))
    ax.set_yticklabels(split_names)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color="white" if data[i, j] < 0.65 else "black", fontsize=8)
    ax.set_title("Best Hidden Probe AUROC by Split and Pooling")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    savefig(out_dir / "split_pooling_best_auroc_heatmap.png")


def plot_score_distributions(meta, scores, y, score_names, out_dir):
    fig, axes = plt.subplots(1, len(score_names), figsize=(4 * len(score_names), 3.6))
    if len(score_names) == 1:
        axes = [axes]
    for ax, name, col in zip(axes, score_names, range(len(score_names))):
        safe = scores[y == 0, col]
        unsafe = scores[y == 1, col]
        ax.boxplot([safe, unsafe], labels=["safe", "unsafe"], showfliers=False)
        ax.set_title(name)
        ax.set_ylabel("first-token probability")
    fig.suptitle("Generation-Prefix Score Distributions")
    savefig(out_dir / "prefix_score_distributions.png")

    by_source = defaultdict(lambda: [0, 0])
    for row, label in zip(meta, y):
        key = str(row.get("source") or "None")
        by_source[key][1] += 1
        by_source[key][0] += int(label)
    names = sorted(by_source)
    rates = [by_source[name][0] / by_source[name][1] for name in names]
    plt.figure(figsize=(9, 4))
    plt.bar(names, rates)
    plt.ylabel("Unsafe rate")
    plt.title("Unsafe Rate by Source")
    plt.xticks(rotation=35, ha="right")
    savefig(out_dir / "unsafe_rate_by_source.png")


def plot_threshold_tradeoff(best, out_dir):
    rows = best["threshold_metrics"]
    thresholds = [r["threshold"] for r in rows]
    plt.figure(figsize=(7, 4.2))
    for key, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("route_rate", "Route rate"),
        ("residual_unsafe_rate", "Residual unsafe"),
    ]:
        plt.plot(thresholds, [r[key] for r in rows], marker="o", label=label)
    plt.xlabel("Probe threshold")
    plt.ylabel("Rate")
    plt.ylim(0.0, 1.0)
    plt.title("Router Threshold Trade-off")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    savefig(out_dir / "best_probe_threshold_tradeoff.png")


def plot_pca(x, y, meta, best, out_dir):
    coords = pca2(x)
    plt.figure(figsize=(6.2, 5.2))
    colors = np.where(y == 1, "#b23b3b", "#2f5d9b")
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA: {best['pooling']} layer {best['layer']} hidden states")
    savefig(out_dir / "best_hidden_pca_safe_vs_unsafe.png")

    sources = sorted({str(row.get("source") or "None") for row in meta})
    color_map = {name: plt.cm.tab10(i % 10) for i, name in enumerate(sources)}
    plt.figure(figsize=(6.8, 5.4))
    for source in sources:
        idx = [i for i, row in enumerate(meta) if str(row.get("source") or "None") == source]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=16, alpha=0.70, label=source, color=color_map[source])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Colored by AudioJailbreak Source")
    plt.legend(frameon=False, fontsize=8)
    savefig(out_dir / "best_hidden_pca_by_source.png")


def write_summary(path, results, best, meta, y):
    lines = ["# AudioJailbreak Hidden-State Linear Probe\n"]
    lines.append(f"- rows: {len(meta)}")
    lines.append(f"- unsafe positives: {int(y.sum())} / {len(y)} ({float(y.mean()) if len(y) else 0.0:.3f})")
    lines.append("")
    lines.append("## Best Random-Split Probe\n")
    lines.append(f"- feature set: `{best['feature_set']}`")
    lines.append(f"- pooling: `{best.get('pooling')}`")
    lines.append(f"- layer: `{best.get('layer')}`")
    lines.append(f"- AUROC: `{best['auroc']:.4f}`")
    lines.append(f"- AUPRC: `{best['auprc']:.4f}`")
    lines.append("")
    lines.append("## Top Layer-Wise Hidden Probes\n")
    lines.append("| Split | Pooling | Layer | AUROC | AUPRC | Positives/Test |")
    lines.append("|---|---|---:|---:|---:|---:|")
    top = sorted(
        [r for r in results if r["feature_set"] == "hidden"],
        key=lambda r: (r["split"] != "random", -r["auroc"]),
    )[:20]
    for row in top:
        lines.append(
            f"| {row['split']} | {row['pooling']} | {row['layer']} | "
            f"{row['auroc']:.4f} | {row['auprc']:.4f} | {row['test_positives']} / {row['test_rows']} |"
        )
    lines.append("")
    lines.append("## Feature-Set Comparison\n")
    lines.append("| Split | Feature Set | Pooling | Layer | AUROC | AUPRC |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in sorted(results, key=lambda r: (r["split"], r["feature_set"], -r["auroc"]))[:80]:
        if row["feature_set"] in {"score_only", "hidden_plus_scores", "concat_layers"}:
            lines.append(
                f"| {row['split']} | {row['feature_set']} | {row.get('pooling')} | "
                f"{row.get('layer', '')} | {row['auroc']:.4f} | {row['auprc']:.4f} |"
            )
    lines.append("")
    lines.append("## Figures\n")
    for name in [
        "layerwise_auroc_by_pooling.png",
        "split_pooling_best_auroc_heatmap.png",
        "prefix_score_distributions.png",
        "unsafe_rate_by_source.png",
        "best_probe_threshold_tradeoff.png",
        "best_hidden_pca_safe_vs_unsafe.png",
        "best_hidden_pca_by_source.png",
    ]:
        lines.append(f"- `{name}`")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--max-heldout-splits", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.features, allow_pickle=True)
    meta = load_jsonl(args.meta)
    y = data["labels"].astype(int)
    scores = data["scores"].astype(np.float32)
    score_names = [str(x) for x in data["score_names"].tolist()]
    pool_names = [str(x) for x in data["pool_names"].tolist()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    valid_idx = np.asarray([i for i, row in enumerate(meta) if not row.get("extract_error")], dtype=int)
    y = y[valid_idx]
    scores = scores[valid_idx]
    meta = [meta[i] for i in valid_idx.tolist()]

    random_train, random_test = stratified_random_split(y, args.train_frac, args.seed)
    splits = [("random", random_train, random_test)]
    splits.extend(heldout_splits(meta, "source", y)[: args.max_heldout_splits])
    splits.extend(heldout_splits(meta, "category", y)[: args.max_heldout_splits])

    results = []
    best_model = None
    best_probs = None
    best_x = None

    def evaluate(name, x, split_name, train_idx, test_idx, feature_set, pooling=None, layer=None):
        nonlocal best_model, best_probs, best_x
        probs, model_info = fit_predict(x[train_idx], y[train_idx], x[test_idx], args.seed)
        result = {
            "split": split_name,
            "name": name,
            "feature_set": feature_set,
            "pooling": pooling,
            "layer": int(layer) if layer is not None else None,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_positives": int(y[train_idx].sum()),
            "test_positives": int(y[test_idx].sum()),
            "auroc": auroc(y[test_idx], probs),
            "auprc": auprc(y[test_idx], probs),
        }
        if split_name == "random":
            result["threshold_metrics"] = threshold_metrics(y[test_idx], probs, thresholds)
        results.append(result)
        if split_name == "random" and feature_set == "hidden" and (best_model is None or result["auroc"] > best_model["auroc"]):
            best_model = dict(result)
            best_model["model_info"] = {
                "mean": model_info["mean"].tolist(),
                "std": model_info["std"].tolist(),
                "coef": model_info["coef"].tolist(),
                "bias": float(model_info["bias"]),
            }
            best_probs = probs
            best_x = x.copy()

    print("[INFO] valid rows:", len(meta))
    print("[INFO] label counts:", Counter(y.tolist()))
    print("[INFO] splits:", [name for name, _, _ in splits])

    for split_name, train_idx, test_idx in splits:
        print(f"[INFO] probing split={split_name} train={len(train_idx)} test={len(test_idx)}", flush=True)
        evaluate("score_only", scores, split_name, train_idx, test_idx, "score_only")
        for pool in pool_names:
            print(f"[INFO]  pool={pool}", flush=True)
            hidden = data[f"hidden_{pool}"][valid_idx].astype(np.float32)
            positions = layer_positions(data, f"hidden_{pool}")
            for item in positions:
                evaluate(
                    f"{pool}_layer{item['layer']}",
                    hidden[:, item["layer_pos"], :],
                    split_name,
                    train_idx,
                    test_idx,
                    "hidden",
                    pool,
                    item["layer"],
                )
            selected_layers = selected_position_layers(data, f"hidden_{pool}")
            concat = np.concatenate([hidden[:, item["layer_pos"], :] for item in selected_layers], axis=1)
            evaluate(f"{pool}_concat", concat, split_name, train_idx, test_idx, "concat_layers", pool, -1)
            best_layer = max(
                [r for r in results if r["split"] == split_name and r["feature_set"] == "hidden" and r["pooling"] == pool],
                key=lambda r: r["auroc"],
            )["layer"]
            best_layer_pos = {item["layer"]: item["layer_pos"] for item in positions}[best_layer]
            evaluate(
                f"{pool}_layer{best_layer}_scores",
                np.concatenate([hidden[:, best_layer_pos, :], scores], axis=1),
                split_name,
                train_idx,
                test_idx,
                "hidden_plus_scores",
                pool,
                best_layer,
            )

    best_model["threshold_metrics"] = threshold_metrics(y[random_test], best_probs, thresholds)
    serializable_results = []
    for row in results:
        item = {k: v for k, v in row.items() if k != "threshold_metrics"}
        if "threshold_metrics" in row:
            item["threshold_metrics"] = row["threshold_metrics"]
        serializable_results.append(item)

    result_json = {
        "features": args.features,
        "meta": args.meta,
        "rows": len(meta),
        "label_counts": dict(Counter(y.tolist())),
        "score_names": score_names,
        "pool_names": pool_names,
        "splits": [{"name": name, "train": len(train), "test": len(test)} for name, train, test in splits],
        "best_random_hidden_probe": best_model,
        "results": serializable_results,
    }
    (out_dir / "probe_results.json").write_text(json.dumps(result_json, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_layerwise(results, out_dir)
    plot_split_heatmap(results, out_dir)
    plot_score_distributions(meta, scores, y, score_names, out_dir)
    plot_threshold_tradeoff(best_model, out_dir)
    plot_pca(best_x, y, meta, best_model, out_dir)
    write_summary(out_dir / "probe_summary.md", results, best_model, meta, y)

    print("[OK] wrote", out_dir / "probe_results.json")
    print("[OK] wrote", out_dir / "probe_summary.md")
    print("[INFO] label counts:", Counter(y.tolist()))
    print("[INFO] best random hidden:", best_model["pooling"], best_model["layer"], best_model["auroc"])


if __name__ == "__main__":
    main()
