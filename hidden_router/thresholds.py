import numpy as np

from hidden_router.metrics import confusion_counts


def threshold_metrics(y, scores, threshold):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    counts = confusion_counts(y, pred)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    route_rate = float(pred.mean()) if len(pred) else 0.0
    safe_total = int((y == 0).sum())
    before_unsafe = float(y.mean()) if len(y) else 0.0
    after_unsafe = fn / len(y) if len(y) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "route_rate": route_rate,
        "safe_false_route": fp / safe_total if safe_total else 0.0,
        "residual_unsafe_rate": after_unsafe,
        "unsafe_before": before_unsafe,
        "unsafe_after": after_unsafe,
        "relative_reduction": (before_unsafe - after_unsafe) / before_unsafe if before_unsafe else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_sweep_metrics(y, scores, thresholds):
    return [threshold_metrics(y, scores, threshold) for threshold in thresholds]


def choose_threshold(y_val, p_val, objective):
    best = None
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = threshold_metrics(y_val, p_val, threshold)
        if objective == "high_recall":
            score = (metrics["recall"] >= 0.80, metrics["f1"], -metrics["route_rate"], metrics["recall"])
        elif objective == "low_route":
            score = (metrics["f1"], -metrics["route_rate"], metrics["recall"])
        else:
            score = (metrics["f1"], metrics["recall"], -metrics["route_rate"])
        row = {
            "threshold": float(threshold),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "route_rate": metrics["route_rate"],
            "val_metrics": metrics,
            "score": score,
        }
        if best is None or row["score"] > best["score"]:
            best = row
    best.pop("score", None)
    return best


def threshold_for_route_rate(scores, route_rate):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 1.0
    q = max(0.0, min(1.0, 1.0 - route_rate))
    return float(np.quantile(scores, q))

