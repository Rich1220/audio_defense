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

from layer_utils import layer_positions, layer_regions as mapped_layer_regions


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


def stratified_split_indices(indices, y, train_frac, seed):
    rng = random.Random(seed)
    pos = [i for i in indices if int(y[i]) == 1]
    neg = [i for i in indices if int(y[i]) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = int(round(len(pos) * train_frac))
    n_neg = int(round(len(neg) * train_frac))
    train = pos[:n_pos] + neg[:n_neg]
    test = pos[n_pos:] + neg[n_neg:]
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def heldout_splits(meta, key, y, max_splits):
    groups = defaultdict(list)
    for i, row in enumerate(meta):
        groups[str(row.get(key) or "None")].append(i)
    all_idx = set(range(len(meta)))
    splits = []
    for value, test in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
        test_idx = np.asarray(test, dtype=int)
        train_idx = np.asarray(sorted(all_idx - set(test)), dtype=int)
        if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
            continue
        if y[test_idx].sum() < 2 or y[train_idx].sum() < 2:
            continue
        splits.append((f"{key}={value}", train_idx, test_idx))
        if len(splits) >= max_splits:
            break
    return splits


def layer_regions(num_layers):
    rows = []
    for name, start_frac, end_frac in REGIONS:
        start = int(np.floor(start_frac * num_layers))
        end = int(np.floor(end_frac * num_layers))
        if name == REGIONS[-1][0]:
            end = num_layers
        end = max(end, start + 1)
        start = min(start, num_layers - 1)
        end = min(end, num_layers)
        rows.append((name, start, end))
    return rows


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


def threshold_metrics(y, scores, threshold):
    y = np.asarray(y).astype(int)
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    route_rate = float(pred.mean()) if len(pred) else 0.0
    false_route = fp / int((y == 0).sum()) if int((y == 0).sum()) else 0.0
    before_unsafe = float(y.mean()) if len(y) else 0.0
    after_unsafe = fn / len(y) if len(y) else 0.0
    reduction = (before_unsafe - after_unsafe) / before_unsafe if before_unsafe else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "route_rate": route_rate,
        "safe_false_route": false_route,
        "unsafe_before": before_unsafe,
        "unsafe_after": after_unsafe,
        "relative_reduction": reduction,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def choose_threshold(y_val, p_val, objective):
    best = None
    for threshold in np.linspace(0.05, 0.95, 19):
        m = threshold_metrics(y_val, p_val, threshold)
        if objective == "high_recall":
            score = (m["recall"] >= 0.80, m["f1"], -m["route_rate"], m["recall"])
        elif objective == "low_route":
            score = (m["f1"], -m["route_rate"], m["recall"])
        else:
            score = (m["f1"], m["recall"], -m["route_rate"])
        if best is None or score > best["score"]:
            best = {"threshold": float(threshold), "score": score, "val_metrics": m}
    best.pop("score", None)
    return best


def get_x(data, pool_key, valid_idx, layer_pos):
    return data[pool_key][valid_idx, layer_pos, :].astype(np.float32)


def sweep_candidates(data, valid_idx, y, fit_idx, val_idx, seed):
    rows = []
    for pool_name, pool_key in POOL_KEYS:
        for item in layer_positions(data, pool_key):
            x = get_x(data, pool_key, valid_idx, item["layer_pos"])
            model = train_model(x[fit_idx], y[fit_idx], seed)
            p_val = predict_model(model, x[val_idx])
            rows.append(
                {
                    "pooling": pool_name,
                    "pool_key": pool_key,
                    "layer": item["layer"],
                    "layer_pos": item["layer_pos"],
                    "val_auroc": auroc(y[val_idx], p_val),
                    "val_auprc": auprc(y[val_idx], p_val),
                }
            )
    return rows


def select_by_depth_regions(sweep_rows, data, metric):
    selected = []
    key = "val_auprc" if metric == "auprc" else "val_auroc"
    # Regions are defined over available layer positions, while reported
    # region_start/end and layer values use real model layer ids.
    for region in mapped_layer_regions(data, POOL_KEYS[0][1], REGIONS):
        candidates = [
            r
            for r in sweep_rows
            if region["start_pos"] <= r["layer_pos"] < region["end_pos"]
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: (np.nan_to_num(r[key], nan=-1.0), np.nan_to_num(r["val_auprc"], nan=-1.0)))
        item = dict(best)
        item["region"] = region["region"]
        item["region_start"] = region["region_start"]
        item["region_end"] = region["region_end"]
        selected.append(item)
    return selected


def select_topk(sweep_rows, k, metric):
    key = "val_auprc" if metric == "auprc" else "val_auroc"
    ranked = sorted(
        sweep_rows,
        key=lambda r: (np.nan_to_num(r[key], nan=-1.0), np.nan_to_num(r["val_auprc"], nan=-1.0)),
        reverse=True,
    )
    selected = []
    for i, row in enumerate(ranked[:k]):
        item = dict(row)
        item["region"] = f"top{i + 1}"
        item["region_start"] = None
        item["region_end"] = None
        selected.append(item)
    return selected


def run_ensemble(data, valid_idx, y, train_idx, test_idx, selected, seed, objective):
    fit_idx, val_idx = stratified_split_indices(train_idx.tolist(), y, 0.80, seed)
    val_probs = []
    test_probs = []
    models = []
    for item in selected:
        x = get_x(data, item["pool_key"], valid_idx, item["layer_pos"])
        fit_model = train_model(x[fit_idx], y[fit_idx], seed)
        val_probs.append(predict_model(fit_model, x[val_idx]))
        final_model = train_model(x[train_idx], y[train_idx], seed)
        test_probs.append(predict_model(final_model, x[test_idx]))
        models.append(
            {
                "region": item["region"],
                "pooling": item["pooling"],
                "pool_key": item["pool_key"],
                "layer": item["layer"],
                "val_auroc": item["val_auroc"],
                "val_auprc": item["val_auprc"],
                "mean": final_model["mean"].tolist(),
                "std": final_model["std"].tolist(),
                "coef": final_model["coef"].tolist(),
                "bias": final_model["bias"],
            }
        )
    p_val = np.mean(np.vstack(val_probs), axis=0)
    p_test = np.mean(np.vstack(test_probs), axis=0)
    threshold = choose_threshold(y[val_idx], p_val, objective)
    return {
        "risk_scores_test": p_test,
        "threshold_selection": threshold,
        "defense": threshold_metrics(y[test_idx], p_test, threshold["threshold"]),
        "auroc": auroc(y[test_idx], p_test),
        "auprc": auprc(y[test_idx], p_test),
        "models": models,
    }


def run_best_single(data, valid_idx, y, train_idx, test_idx, selected_single, seed, objective):
    fit_idx, val_idx = stratified_split_indices(train_idx.tolist(), y, 0.80, seed)
    x = get_x(data, selected_single["pool_key"], valid_idx, selected_single["layer_pos"])
    fit_model = train_model(x[fit_idx], y[fit_idx], seed)
    p_val = predict_model(fit_model, x[val_idx])
    threshold = choose_threshold(y[val_idx], p_val, objective)
    final_model = train_model(x[train_idx], y[train_idx], seed)
    p_test = predict_model(final_model, x[test_idx])
    return {
        "selected": selected_single,
        "threshold_selection": threshold,
        "defense": threshold_metrics(y[test_idx], p_test, threshold["threshold"]),
        "auroc": auroc(y[test_idx], p_test),
        "auprc": auprc(y[test_idx], p_test),
    }


def run_split(split_name, train_idx, test_idx, data, valid_idx, y, seed, metric, objective, selection_mode, topk):
    fit_idx, val_idx = stratified_split_indices(train_idx.tolist(), y, 0.80, seed)
    sweep_rows = sweep_candidates(data, valid_idx, y, fit_idx, val_idx, seed)
    if selection_mode == "topk":
        selected = select_topk(sweep_rows, topk, metric)
    else:
        selected = select_by_depth_regions(sweep_rows, data, metric)
    key = "val_auprc" if metric == "auprc" else "val_auroc"
    best_single = max(
        sweep_rows,
        key=lambda r: (np.nan_to_num(r[key], nan=-1.0), np.nan_to_num(r["val_auprc"], nan=-1.0)),
    )
    ensemble = run_ensemble(data, valid_idx, y, train_idx, test_idx, selected, seed, objective)
    single = run_best_single(data, valid_idx, y, train_idx, test_idx, best_single, seed, objective)
    return {
        "split": split_name,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "test_positives": int(y[test_idx].sum()),
        "selection_mode": selection_mode,
        "metric": metric,
        "selected_layers": selected,
        "best_single": single,
        "ensemble": {
            "auroc": ensemble["auroc"],
            "auprc": ensemble["auprc"],
            "threshold_selection": ensemble["threshold_selection"],
            "defense": ensemble["defense"],
            "models": ensemble["models"],
        },
        "sweep_rows": sweep_rows,
    }


def write_selected_csv(path, results):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "region",
                "pooling",
                "layer",
                "region_start",
                "region_end",
                "val_auroc",
                "val_auprc",
            ],
        )
        writer.writeheader()
        for result in results:
            for row in result["selected_layers"]:
                writer.writerow(
                    {
                        "split": result["split"],
                        "region": row["region"],
                        "pooling": row["pooling"],
                        "layer": row["layer"],
                        "region_start": row["region_start"],
                        "region_end": row["region_end"],
                        "val_auroc": row["val_auroc"],
                        "val_auprc": row["val_auprc"],
                    }
                )


def write_summary(path, results, label_counts, args):
    lines = ["# AudioJailbreak Auto-Layer Router\n"]
    lines.append(f"- label counts: `{dict(label_counts)}`")
    lines.append(f"- selection mode: `{args.selection_mode}`")
    lines.append(f"- selection metric: `{args.metric}`")
    lines.append(f"- threshold objective: `{args.objective}`")
    lines.append("")
    lines.append("## Method\n")
    lines.append("For each split, the script sweeps all layers and pooling strategies using train/validation data only, selects layers on validation, trains one probe per selected representation, and averages risk scores.")
    lines.append("")
    lines.append("## Ensemble Results\n")
    lines.append("| Split | Test Positives | AUROC | AUPRC | Unsafe Before | Unsafe After | Reduction | Route Rate | False Route | Recall | Threshold |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        d = result["ensemble"]["defense"]
        lines.append(
            f"| {result['split']} | {result['test_positives']} / {result['test_rows']} | "
            f"{result['ensemble']['auroc']:.3f} | {result['ensemble']['auprc']:.3f} | "
            f"{d['unsafe_before']:.3f} | {d['unsafe_after']:.3f} | {d['relative_reduction']:.3f} | "
            f"{d['route_rate']:.3f} | {d['safe_false_route']:.3f} | {d['recall']:.3f} | {d['threshold']:.2f} |"
        )
    lines.append("")
    lines.append("## Best Single-Layer Baseline\n")
    lines.append("| Split | Pooling | Layer | Val AUROC | Val AUPRC | Test AUROC | Test AUPRC | Unsafe After | Route Rate | Recall |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        s = result["best_single"]
        d = s["defense"]
        selected = s["selected"]
        lines.append(
            f"| {result['split']} | {selected['pooling']} | {selected['layer']} | "
            f"{selected['val_auroc']:.3f} | {selected['val_auprc']:.3f} | "
            f"{s['auroc']:.3f} | {s['auprc']:.3f} | {d['unsafe_after']:.3f} | {d['route_rate']:.3f} | {d['recall']:.3f} |"
        )
    lines.append("")
    lines.append("## Selected Layers\n")
    lines.append("| Split | Region | Pooling | Layer | Region Layers | Val AUROC | Val AUPRC |")
    lines.append("|---|---|---|---:|---|---:|---:|")
    for result in results:
        for row in result["selected_layers"]:
            region_layers = f"{row['region_start']}-{row['region_end']}" if row["region_start"] is not None else "-"
            lines.append(
                f"| {result['split']} | {row['region']} | {row['pooling']} | {row['layer']} | "
                f"{region_layers} | {row['val_auroc']:.3f} | {row['val_auprc']:.3f} |"
            )
    lines.append("")
    lines.append("## Figures\n")
    lines.append("- `auto_layer_ensemble_before_after.png`")
    lines.append("- `auto_layer_route_tradeoff.png`")
    lines.append("- `auto_layer_auroc_by_split.png`")
    lines.append("- `auto_layer_selected_layers.png`")
    lines.append("- `auto_layer_random_sweep_heatmap.png`")
    lines.append("- `sweep_heatmaps/` for selected split-level layer sweep heatmaps")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plot_before_after(results, out_dir):
    names = [r["split"] for r in results]
    before = [r["ensemble"]["defense"]["unsafe_before"] for r in results]
    after = [r["ensemble"]["defense"]["unsafe_after"] for r in results]
    x = np.arange(len(names))
    plt.figure(figsize=(max(8, 0.55 * len(names)), 4.5))
    plt.bar(x - 0.18, before, width=0.36, label="Before")
    plt.bar(x + 0.18, after, width=0.36, label="After")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("Unsafe rate")
    plt.title("Auto-Layer Ensemble Router Simulation")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = out_dir / "auto_layer_ensemble_before_after.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", path)


def plot_selected_layers(results, out_dir):
    pool_colors = {"mean_context": "#4267a8", "audio_tokens": "#d9792b", "last": "#4d8f3a"}
    region_markers = {"shallow": "o", "middle": "s", "deep": "^", "final": "D"}
    plt.figure(figsize=(max(8, 0.7 * len(results)), 4.8))
    for i, result in enumerate(results):
        for row in result["selected_layers"]:
            plt.scatter(
                i,
                row["layer"],
                s=70,
                color=pool_colors.get(row["pooling"], "#555555"),
                marker=region_markers.get(row["region"], "o"),
                alpha=0.9,
            )
    plt.xticks(range(len(results)), [r["split"] for r in results], rotation=35, ha="right")
    plt.ylabel("Selected layer")
    plt.title("Auto-Selected Layers by Split")
    handles = []
    labels = []
    for pool, color in pool_colors.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8))
        labels.append(pool)
    plt.legend(handles, labels, frameon=False, loc="best")
    plt.tight_layout()
    path = out_dir / "auto_layer_selected_layers.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", path)


def plot_route_tradeoff(results, out_dir):
    names = [r["split"] for r in results]
    route = [r["ensemble"]["defense"]["route_rate"] for r in results]
    recall = [r["ensemble"]["defense"]["recall"] for r in results]
    false_route = [r["ensemble"]["defense"]["safe_false_route"] for r in results]
    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(route, recall, s=60, label="Caught unsafe recall")
    plt.scatter(route, false_route, s=60, marker="x", label="Safe false-route")
    for name, x, yv in zip(names, route, recall):
        if name in {"random", "source=PAIR", "source=GCG"}:
            plt.annotate(name, (x, yv), textcoords="offset points", xytext=(5, 4), fontsize=8)
    plt.xlabel("Route rate")
    plt.ylabel("Rate")
    plt.title("Auto-Layer Router Safety-Utility Tradeoff")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    path = out_dir / "auto_layer_route_tradeoff.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", path)


def plot_auroc_by_split(results, out_dir):
    names = [r["split"] for r in results]
    ensemble_auroc = [r["ensemble"]["auroc"] for r in results]
    single_auroc = [r["best_single"]["auroc"] for r in results]
    x = np.arange(len(names))
    plt.figure(figsize=(max(9, 0.6 * len(names)), 4.8))
    plt.bar(x - 0.18, single_auroc, width=0.36, label="Best single layer")
    plt.bar(x + 0.18, ensemble_auroc, width=0.36, label="Auto-layer ensemble")
    plt.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Test AUROC")
    plt.title("Best Single vs Auto-Layer Ensemble")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = out_dir / "auto_layer_auroc_by_split.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", path)


def plot_random_heatmap(results, out_dir):
    random_rows = [r for r in results if r["split"] == "random"]
    if not random_rows:
        return
    sweep = random_rows[0]["sweep_rows"]
    layers = sorted({r["layer"] for r in sweep})
    pools = [name for name, _ in POOL_KEYS]
    data = np.full((len(pools), len(layers)), np.nan)
    for r in sweep:
        data[pools.index(r["pooling"]), layers.index(r["layer"])] = r["val_auroc"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(pools)))
    ax.set_yticklabels(pools)
    ax.set_xticks(np.arange(0, len(layers), max(1, len(layers) // 12)))
    ax.set_xticklabels([str(layers[i]) for i in np.arange(0, len(layers), max(1, len(layers) // 12))])
    ax.set_xlabel("Layer")
    ax.set_title("Random Split Validation AUROC Sweep")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    path = out_dir / "auto_layer_random_sweep_heatmap.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", path)


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)[:120]


def plot_sweep_heatmap_for_split(result, out_dir):
    sweep = result["sweep_rows"]
    layers = sorted({r["layer"] for r in sweep})
    pools = [name for name, _ in POOL_KEYS]
    data = np.full((len(pools), len(layers)), np.nan)
    for r in sweep:
        data[pools.index(r["pooling"]), layers.index(r["layer"])] = r["val_auroc"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(pools)))
    ax.set_yticklabels(pools)
    tick_idx = np.arange(0, len(layers), max(1, len(layers) // 12))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(layers[i]) for i in tick_idx])
    ax.set_xlabel("Layer")
    ax.set_title(f"Validation AUROC Sweep: {result['split']}")
    for row in result["selected_layers"]:
        if row["layer"] in layers and row["pooling"] in pools:
            ax.scatter(layers.index(row["layer"]), pools.index(row["pooling"]), s=80, facecolors="none", edgecolors="red", linewidths=1.8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    heatmap_dir = out_dir / "sweep_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    path = heatmap_dir / f"{safe_name(result['split'])}_sweep_heatmap.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", path)


def write_outputs(out_dir, results, y, args, data, meta_path, features_path):
    result_json = {
        "features": features_path,
        "meta": meta_path,
        "rows": int(sum(Counter(y.tolist()).values())),
        "label_counts": dict(Counter(y.tolist())),
        "args": vars(args),
        "regions": mapped_layer_regions(data, POOL_KEYS[0][1], REGIONS),
        "layer_indices": {pool_key: [item["layer"] for item in layer_positions(data, pool_key)] for _, pool_key in POOL_KEYS},
        "pool_keys": POOL_KEYS,
        "results": results,
    }
    (out_dir / "auto_layer_router_results.json").write_text(json.dumps(json_safe(result_json), indent=2, ensure_ascii=False), encoding="utf-8")
    write_selected_csv(out_dir / "auto_layer_selected_layers.csv", results)
    write_summary(out_dir / "auto_layer_router_summary.md", results, Counter(y.tolist()), args)
    if results:
        plot_before_after(results, out_dir)
        plot_route_tradeoff(results, out_dir)
        plot_auroc_by_split(results, out_dir)
        plot_selected_layers(results, out_dir)
        plot_random_heatmap(results, out_dir)
    print("[OK] checkpointed", len(results), "split(s)")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-mode", choices=["all", "random", "source", "category"], default="source")
    parser.add_argument("--selection-mode", choices=["depth_regions", "topk"], default="depth_regions")
    parser.add_argument("--metric", choices=["auroc", "auprc"], default="auroc")
    parser.add_argument("--objective", choices=["f1", "high_recall", "low_route"], default="f1")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-heldout-splits", type=int, default=12)
    parser.add_argument(
        "--plot-sweep-splits",
        default="random,source=PAIR,source=GCG",
        help="Comma-separated split names for per-split sweep heatmaps, or 'all'.",
    )
    parser.add_argument("--checkpoint-every-split", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features, allow_pickle=True)
    meta_all = load_jsonl(args.meta)
    valid_idx = np.asarray([i for i, row in enumerate(meta_all) if not row.get("extract_error")], dtype=int)
    meta = [meta_all[i] for i in valid_idx.tolist()]
    y = data["labels"][valid_idx].astype(int)

    train_idx, test_idx = stratified_split_indices(list(range(len(meta))), y, args.train_frac, args.seed)
    splits = [("random", train_idx, test_idx)]
    if args.split_mode in {"all", "source"}:
        splits.extend(heldout_splits(meta, "source", y, args.max_heldout_splits))
    if args.split_mode in {"all", "category"}:
        splits.extend(heldout_splits(meta, "category", y, args.max_heldout_splits))
    if args.split_mode == "random":
        splits = [("random", train_idx, test_idx)]

    results = []
    requested_heatmaps = {item.strip() for item in args.plot_sweep_splits.split(",") if item.strip()}
    for split_name, train, test in splits:
        print("[INFO] split:", split_name, "train", len(train), "test", len(test), flush=True)
        result = run_split(
            split_name,
            train,
            test,
            data,
            valid_idx,
            y,
            args.seed,
            args.metric,
            args.objective,
            args.selection_mode,
            args.topk,
        )
        results.append(result)
        if args.plot_sweep_splits == "all" or split_name in requested_heatmaps:
            plot_sweep_heatmap_for_split(result, out_dir)
        if args.checkpoint_every_split:
            write_outputs(out_dir, results, y, args, data, args.meta, args.features)

    write_outputs(out_dir, results, y, args, data, args.meta, args.features)
    print("[OK] wrote", out_dir / "auto_layer_router_results.json")
    print("[OK] wrote", out_dir / "auto_layer_selected_layers.csv")
    print("[OK] wrote", out_dir / "auto_layer_router_summary.md")


if __name__ == "__main__":
    main()
