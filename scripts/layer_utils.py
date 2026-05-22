import numpy as np


LAYER_INDEX_KEYS = (
    "layer_indices",
    "hidden_layer_indices",
    "layers",
)


def layer_indices_for(data, pool_key):
    """Return real model layer ids for the hidden layer axis.

    Feature files may contain only a subset of layers, e.g. model layers
    [0, 8, 16, 24, 32]. In that case the hidden array axis is positional,
    but reports and selection logic should use the real layer ids.
    """
    n_layers = int(data[pool_key].shape[1])
    pool_specific_keys = (
        f"{pool_key}_layer_indices",
        f"{pool_key}_layers",
    )
    for key in pool_specific_keys + LAYER_INDEX_KEYS:
        if key in data.files:
            values = np.asarray(data[key]).astype(int).reshape(-1)
            if len(values) != n_layers:
                raise ValueError(
                    f"{key} length {len(values)} != {pool_key} layer axis {n_layers}"
                )
            if len(set(values.tolist())) != len(values):
                raise ValueError(f"{key} contains duplicate layer ids: {values.tolist()}")
            return values.tolist()
    return list(range(n_layers))


def layer_positions(data, pool_key):
    layer_ids = layer_indices_for(data, pool_key)
    return [
        {
            "layer_pos": pos,
            "layer": int(layer_id),
        }
        for pos, layer_id in enumerate(layer_ids)
    ]


def selected_position_layers(data, pool_key):
    positions = layer_positions(data, pool_key)
    n_layers = len(positions)
    selected_pos = sorted(set([0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]))
    return [positions[pos] for pos in selected_pos]


def layer_regions(data, pool_key, regions):
    positions = layer_positions(data, pool_key)
    n_layers = len(positions)
    out = []
    for name, start_frac, end_frac in regions:
        start = int(np.floor(start_frac * n_layers))
        end = int(np.floor(end_frac * n_layers))
        if name == regions[-1][0]:
            end = n_layers
        end = max(end, start + 1)
        start = min(start, n_layers - 1)
        end = min(end, n_layers)
        region_positions = positions[start:end]
        out.append(
            {
                "region": name,
                "start_pos": start,
                "end_pos": end,
                "region_start": region_positions[0]["layer"],
                "region_end": region_positions[-1]["layer"],
                "layer_positions": region_positions,
            }
        )
    return out
