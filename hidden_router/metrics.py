import numpy as np


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


def confusion_counts(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
    }

