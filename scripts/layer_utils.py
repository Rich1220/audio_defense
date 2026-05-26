import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hidden_router.layers import (
    layer_indices_for,
    layer_positions,
    layer_regions,
    selected_position_layers,
)
