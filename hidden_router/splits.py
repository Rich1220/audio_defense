import random
from collections import defaultdict

import numpy as np


def stratified_split_indices(indices, y, train_frac, seed, keep_both_classes=False):
    rng = random.Random(seed)
    pos = [i for i in indices if int(y[i]) == 1]
    neg = [i for i in indices if int(y[i]) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    if keep_both_classes:
        n_pos = max(1, int(round(len(pos) * train_frac))) if len(pos) > 1 else len(pos)
        n_neg = max(1, int(round(len(neg) * train_frac))) if len(neg) > 1 else len(neg)
        n_pos = min(n_pos, len(pos) - 1) if len(pos) > 1 else n_pos
        n_neg = min(n_neg, len(neg) - 1) if len(neg) > 1 else n_neg
    else:
        n_pos = int(round(len(pos) * train_frac))
        n_neg = int(round(len(neg) * train_frac))
    train = pos[:n_pos] + neg[:n_neg]
    test = pos[n_pos:] + neg[n_neg:]
    rng.shuffle(train)
    rng.shuffle(test)
    return np.asarray(train, dtype=int), np.asarray(test, dtype=int)


def stratified_random_split(y, train_frac, seed):
    return stratified_split_indices(list(range(len(y))), y, train_frac, seed)


def heldout_splits(meta, key, y, max_splits=None, min_test_pos=2, min_train_pos=2, sort_by_size=False):
    groups = defaultdict(list)
    for i, row in enumerate(meta):
        groups[str(row.get(key) or "None")].append(i)
    all_idx = set(range(len(meta)))
    items = groups.items()
    if sort_by_size:
        items = sorted(items, key=lambda item: len(item[1]), reverse=True)
    else:
        items = sorted(items)
    splits = []
    for value, test in items:
        test_idx = np.asarray(test, dtype=int)
        train_idx = np.asarray(sorted(all_idx - set(test)), dtype=int)
        if y[test_idx].sum() < min_test_pos or y[train_idx].sum() < min_train_pos:
            continue
        if len(set(y[test_idx].tolist())) < 2 or len(set(y[train_idx].tolist())) < 2:
            continue
        splits.append((f"{key}={value}", train_idx, test_idx))
        if max_splits is not None and len(splits) >= max_splits:
            break
    return splits

