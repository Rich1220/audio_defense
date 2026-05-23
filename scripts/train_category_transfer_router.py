#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from layer_utils import layer_regions as mapped_layer_regions


POOL_KEYS = [
    ("mean_context", "hidden_mean_context"),
    ("audio_tokens", "hidden_audio_tokens"),
    ("last", "hidden_last"),
]

REGIONS = [
    ("shallow", 0.00, 0.25),
    ("middle", 0.25, 0.60),
    ("deep", 0.60, 0.90),
    ("final", 0.90, 1.00),
]


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


def stratified_split(indices, y, train_frac, seed):
    rng = random.Random(seed)
    pos = [i for i in indices if int(y[i]) == 1]
    neg = [i for i in indices if int(y[i]) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = max(1, int(round(len(pos) * train_frac))) if len(pos) > 1 else len(pos)
    n_neg = max(1, int(round(len(neg) * train_frac))) if len(neg) > 1 else len(neg)
    n_pos = min(n_pos, len(pos) - 1) if len(pos) > 1 else n_pos
    n_neg = min(n_neg, len(neg) - 1) if len(neg) > 1 else n_neg
    train = pos[:n_pos] + neg[:n_neg]
    test = pos[n_pos:] + neg[n_neg:]
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def train_model(x_train, y_train, seed):
    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=500,
        random_state=seed,
    )
    clf.fit(x_train_z, y_train.astype(int))
    return {
        "mean": scaler.mean_.astype(float),
        "std": scaler.scale_.astype(float),
        "coef": clf.coef_[0].astype(float),
        "bias": float(clf.intercept_[0]),
    }


def predict_model(model, x):
    x_z = (x - model["mean"]) / np.maximum(model["std"], 1e-8)
    return sigmoid(x_z @ model["coef"] + model["bias"])


def threshold_for_route_rate(scores, route_rate):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 1.0
    q = max(0.0, min(1.0, 1.0 - route_rate))
    return float(np.quantile(scores, q))


def metrics_at_threshold(y, scores, threshold, budget_route_rate):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    before = float(y.mean()) if len(y) else 0.0
    after = fn / len(y) if len(y) else 0.0
    budget_threshold = threshold_for_route_rate(scores, budget_route_rate)
    budget_pred = (scores >= budget_threshold).astype(int)
    budget_tp = int(((budget_pred == 1) & (y == 1)).sum())
    budget_fp = int(((budget_pred == 1) & (y == 0)).sum())
    return {
        "n": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "auroc": auroc(y, scores),
        "auprc": auprc(y, scores),
        "threshold": float(threshold),
        "route_rate": float(pred.mean()) if len(pred) else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_route_rate": fp / negatives if negatives else 0.0,
        "unsafe_before": before,
        "unsafe_after": after,
        "relative_reduction": (before - after) / before if before else 0.0,
        "budget_route_rate": float(budget_pred.mean()) if len(budget_pred) else 0.0,
        "budget_recall": budget_tp / positives if positives else 0.0,
        "budget_false_route_rate": budget_fp / negatives if negatives else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def get_x(data, pool_key, valid_idx, layer):
    return data[pool_key][:, layer, :].astype(np.float32)


def sweep_layers(raw_data, data, valid_idx, y, fit_idx, val_idx, seed):
    rows = []
    for pooling, pool_key in POOL_KEYS:
        for layer_info in mapped_layer_regions(raw_data, pool_key, [("all", 0.0, 1.0)])[0]["layer_positions"]:
            layer_pos = layer_info["layer_pos"]
            layer = layer_info["layer"]
            x = get_x(data, pool_key, valid_idx, layer_pos)
            model = train_model(x[fit_idx], y[fit_idx], seed)
            p_val = predict_model(model, x[val_idx])
            rows.append(
                {
                    "pooling": pooling,
                    "pool_key": pool_key,
                    "layer_pos": layer_pos,
                    "layer": layer,
                    "val_auroc": auroc(y[val_idx], p_val),
                    "val_auprc": auprc(y[val_idx], p_val),
                }
            )
    return rows


def select_depth_regions(raw_data, sweep_rows, metric):
    selected = []
    key = "val_auprc" if metric == "auprc" else "val_auroc"
    region_defs = mapped_layer_regions(raw_data, POOL_KEYS[0][1], REGIONS)
    for region_def in region_defs:
        layer_positions = {item["layer_pos"] for item in region_def["layer_positions"]}
        candidates = [r for r in sweep_rows if r["layer_pos"] in layer_positions]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: (np.nan_to_num(r[key], nan=-1.0), np.nan_to_num(r["val_auprc"], nan=-1.0)))
        item = dict(best)
        item["region"] = region_def["region"]
        item["region_start"] = region_def["region_start"]
        item["region_end"] = region_def["region_end"]
        selected.append(item)
    return selected


def fit_ensemble(data, valid_idx, y, train_idx, val_idx, selected, seed):
    models = []
    val_probs = []
    for item in selected:
        x = get_x(data, item["pool_key"], valid_idx, item["layer_pos"])
        model = train_model(x[train_idx], y[train_idx], seed)
        val_probs.append(predict_model(model, x[val_idx]))
        models.append(
            {
                "region": item["region"],
                "pooling": item["pooling"],
                "pool_key": item["pool_key"],
                "layer_pos": item["layer_pos"],
                "layer": item["layer"],
                "val_auroc": item["val_auroc"],
                "val_auprc": item["val_auprc"],
                "model": model,
            }
        )
    return models, np.mean(np.vstack(val_probs), axis=0)


def predict_ensemble(data, valid_idx, models, indices):
    probs = []
    for item in models:
        x = get_x(data, item["pool_key"], valid_idx, item["layer_pos"])
        probs.append(predict_model(item["model"], x[indices]))
    return np.mean(np.vstack(probs), axis=0)


def category_counts(meta, y):
    counts = defaultdict(Counter)
    for i, row in enumerate(meta):
        counts[str(row.get("category"))][int(y[i])] += 1
    return counts


def plot_matrix(rows, categories, metric, out_path, title):
    matrix = np.full((len(categories), len(categories)), np.nan)
    for row in rows:
        if row["train_category"] in categories and row["test_category"] in categories:
            i = categories.index(row["train_category"])
            j = categories.index(row["test_category"])
            matrix[i, j] = row[metric]
    fig, ax = plt.subplots(figsize=(max(7.5, 0.75 * len(categories)), max(6.5, 0.65 * len(categories))))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_xlabel("Test category")
    ax.set_ylabel("Train category")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", out_path)


def write_csv(path, rows):
    if not rows:
        return
    fields = [
        "train_category",
        "test_category",
        "test_scope",
        "n",
        "positives",
        "negatives",
        "auroc",
        "auprc",
        "threshold",
        "route_rate",
        "precision",
        "recall",
        "false_route_rate",
        "budget_route_rate",
        "budget_recall",
        "budget_false_route_rate",
        "unsafe_before",
        "unsafe_after",
        "relative_reduction",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_summary(path, rows, train_categories, selected_by_train, args):
    lines = ["# Category Transfer Router\n"]
    lines.append(f"- features: `{args.features}`")
    lines.append(f"- meta: `{args.meta}`")
    lines.append(f"- min train positives: `{args.min_train_positives}`")
    lines.append(f"- min test positives: `{args.min_test_positives}`")
    lines.append(f"- target validation route rate: `{args.route_rate:.3f}`")
    lines.append(f"- selection metric: `{args.metric}`")
    lines.append("")
    lines.append("This experiment trains an auto-layer hidden router on one harmful category and evaluates it on held-out samples from every eligible category. It tests whether category-specific training learns a general unsafe response-mode signal or a topic-specific shortcut.")
    lines.append("")

    lines.append("## Eligible Train Categories\n")
    for cat in train_categories:
        lines.append(f"- {cat}")
    lines.append("")

    lines.append("## Transfer Matrix: AUROC\n")
    lines.append("| Train \\ Test | " + " | ".join(train_categories) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(train_categories)) + "|")
    by_pair = {(r["train_category"], r["test_category"]): r for r in rows}
    for train_cat in train_categories:
        vals = []
        for test_cat in train_categories:
            row = by_pair.get((train_cat, test_cat))
            vals.append("NA" if row is None or np.isnan(row["auroc"]) else f"{row['auroc']:.3f}")
        lines.append(f"| {train_cat} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Transfer Matrix: Recall at Fixed Validation Route Rate\n")
    lines.append("| Train \\ Test | " + " | ".join(train_categories) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(train_categories)) + "|")
    for train_cat in train_categories:
        vals = []
        for test_cat in train_categories:
            row = by_pair.get((train_cat, test_cat))
            vals.append("NA" if row is None else f"{row['recall']:.3f}")
        lines.append(f"| {train_cat} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append(f"## Transfer Matrix: Recall at {args.route_rate:.0%} Test Budget\n")
    lines.append("| Train \\ Test | " + " | ".join(train_categories) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(train_categories)) + "|")
    for train_cat in train_categories:
        vals = []
        for test_cat in train_categories:
            row = by_pair.get((train_cat, test_cat))
            vals.append("NA" if row is None else f"{row['budget_recall']:.3f}")
        lines.append(f"| {train_cat} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Source Category Strength\n")
    lines.append("| Train Category | Mean Off-Diagonal AUROC | Mean Off-Diagonal Recall | Mean Off-Diagonal Route Rate |")
    lines.append("|---|---:|---:|---:|")
    for train_cat in train_categories:
        off = [r for r in rows if r["train_category"] == train_cat and r["test_category"] != train_cat]
        lines.append(
            f"| {train_cat} | "
            f"{np.nanmean([r['auroc'] for r in off]) if off else float('nan'):.3f} | "
            f"{np.mean([r['recall'] for r in off]) if off else 0.0:.3f} | "
            f"{np.mean([r['route_rate'] for r in off]) if off else 0.0:.3f} |"
        )

    lines.append("")
    lines.append("## Target Category Hardness\n")
    lines.append("| Test Category | Mean Off-Diagonal AUROC | Mean Off-Diagonal Recall | Mean Off-Diagonal Route Rate |")
    lines.append("|---|---:|---:|---:|")
    for test_cat in train_categories:
        off = [r for r in rows if r["test_category"] == test_cat and r["train_category"] != test_cat]
        lines.append(
            f"| {test_cat} | "
            f"{np.nanmean([r['auroc'] for r in off]) if off else float('nan'):.3f} | "
            f"{np.mean([r['recall'] for r in off]) if off else 0.0:.3f} | "
            f"{np.mean([r['route_rate'] for r in off]) if off else 0.0:.3f} |"
        )

    lines.append("")
    lines.append("## Selected Layers by Train Category\n")
    lines.append("| Train Category | Region | Pooling | Layer | Val AUROC | Val AUPRC |")
    lines.append("|---|---|---|---:|---:|---:|")
    for train_cat in train_categories:
        for item in selected_by_train.get(train_cat, []):
            lines.append(
                f"| {train_cat} | {item['region']} | {item['pooling']} | {item['layer']} | "
                f"{item['val_auroc']:.3f} | {item['val_auprc']:.3f} |"
            )

    lines.append("")
    lines.append("## Artifacts\n")
    lines.append("- `category_transfer_results.csv`")
    lines.append("- `category_transfer_selected_layers.json`")
    lines.append("- `category_transfer_auroc_heatmap.png`")
    lines.append("- `category_transfer_recall_heatmap.png`")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items() if k != "model"}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description="Train on one category and evaluate transfer to every eligible category.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-train-positives", type=int, default=10)
    parser.add_argument("--min-test-positives", type=int, default=2)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--fit-frac", type=float, default=0.80)
    parser.add_argument("--route-rate", type=float, default=0.10)
    parser.add_argument("--metric", choices=["auroc", "auprc"], default="auroc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_data = np.load(args.features, allow_pickle=True)
    meta_all = load_jsonl(args.meta)
    valid_idx = np.asarray([i for i, row in enumerate(meta_all) if not row.get("extract_error")], dtype=int)
    meta = [meta_all[i] for i in valid_idx.tolist()]
    y = raw_data["labels"][valid_idx].astype(int)
    data = {
        pool_key: raw_data[pool_key][valid_idx]
        for _, pool_key in POOL_KEYS
    }

    counts = category_counts(meta, y)
    train_categories = [
        cat
        for cat, counter in sorted(counts.items(), key=lambda item: (-item[1][1], item[0]))
        if counter[1] >= args.min_train_positives and counter[0] >= args.min_test_positives
    ]
    category_to_indices = defaultdict(list)
    for i, row in enumerate(meta):
        category_to_indices[str(row.get("category"))].append(i)

    results = []
    selected_by_train = {}
    for train_cat in train_categories:
        source_idx = category_to_indices[train_cat]
        train_pool_idx, same_test_idx = stratified_split(source_idx, y, args.train_frac, args.seed)
        fit_idx, val_idx = stratified_split(train_pool_idx.tolist(), y, args.fit_frac, args.seed)
        if len(set(y[fit_idx].tolist())) < 2 or len(set(y[val_idx].tolist())) < 2:
            print("[WARN] skip train category due to bad split:", train_cat)
            continue
        print("[INFO] train category:", train_cat, "fit", len(fit_idx), "val", len(val_idx), flush=True)
        sweep = sweep_layers(raw_data, data, valid_idx, y, fit_idx, val_idx, args.seed)
        selected = select_depth_regions(raw_data, sweep, args.metric)
        models, val_scores = fit_ensemble(data, valid_idx, y, train_pool_idx, val_idx, selected, args.seed)
        threshold = threshold_for_route_rate(val_scores, args.route_rate)
        selected_by_train[train_cat] = selected

        for test_cat in train_categories:
            if test_cat == train_cat:
                test_idx = same_test_idx
                scope = "same_category_heldout"
            else:
                test_idx = np.asarray(category_to_indices[test_cat], dtype=int)
                scope = "cross_category"
            if int(y[test_idx].sum()) < args.min_test_positives or len(set(y[test_idx].tolist())) < 2:
                continue
            scores = predict_ensemble(data, valid_idx, models, test_idx)
            row = {
                "train_category": train_cat,
                "test_category": test_cat,
                "test_scope": scope,
            }
            row.update(metrics_at_threshold(y[test_idx], scores, threshold, args.route_rate))
            results.append(row)

    write_csv(out_dir / "category_transfer_results.csv", results)
    (out_dir / "category_transfer_selected_layers.json").write_text(
        json.dumps(json_safe(selected_by_train), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(out_dir / "category_transfer_summary.md", results, train_categories, selected_by_train, args)
    plot_matrix(results, train_categories, "auroc", out_dir / "category_transfer_auroc_heatmap.png", "Category Transfer AUROC")
    plot_matrix(results, train_categories, "recall", out_dir / "category_transfer_recall_heatmap.png", f"Category Transfer Recall at {args.route_rate:.0%} Validation Route Rate")
    plot_matrix(results, train_categories, "budget_recall", out_dir / "category_transfer_budget_recall_heatmap.png", f"Category Transfer Recall at {args.route_rate:.0%} Test Budget")
    print("[OK] wrote", out_dir / "category_transfer_summary.md")
    print("[OK] wrote", out_dir / "category_transfer_results.csv")


if __name__ == "__main__":
    main()
