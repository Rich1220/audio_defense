#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python}"
FEATURES="${FEATURES:?Set FEATURES=/path/to/hidden_features.npz}"
META="${META:?Set META=/path/to/hidden_meta.jsonl}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR=/path/to/output_dir}"
SEED="${SEED:-42}"
TRAIN_FRAC="${TRAIN_FRAC:-0.70}"
OBJECTIVE="${OBJECTIVE:-f1}"
SPLIT_MODE="${SPLIT_MODE:-all}"
SELECTION_MODE="${SELECTION_MODE:-depth_regions}"
METRIC="${METRIC:-auroc}"
PLOT_SWEEP_SPLITS="${PLOT_SWEEP_SPLITS:-random}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-hidden-router}"
export MPLCONFIGDIR

mkdir -p "${OUT_DIR}"

echo "[INFO] validating feature schema"
"${PYTHON}" "${ROOT_DIR}/scripts/validate_features.py" \
  --features "${FEATURES}" \
  --meta "${META}"

echo "[INFO] running single-layer probe sweep"
"${PYTHON}" "${ROOT_DIR}/scripts/train_hidden_probes.py" \
  --features "${FEATURES}" \
  --meta "${META}" \
  --out-dir "${OUT_DIR}/single_layer_probe" \
  --seed "${SEED}" \
  --train-frac "${TRAIN_FRAC}"

echo "[INFO] running single-layer router defense simulation"
"${PYTHON}" "${ROOT_DIR}/scripts/simulate_hidden_router_defense.py" \
  --features "${FEATURES}" \
  --meta "${META}" \
  --out-dir "${OUT_DIR}/single_router_defense" \
  --seed "${SEED}" \
  --train-frac "${TRAIN_FRAC}" \
  --objective "${OBJECTIVE}"

echo "[INFO] running model-specific auto-layer router"
"${PYTHON}" "${ROOT_DIR}/scripts/train_auto_layer_router.py" \
  --features "${FEATURES}" \
  --meta "${META}" \
  --out-dir "${OUT_DIR}/auto_layer_router" \
  --split-mode "${SPLIT_MODE}" \
  --selection-mode "${SELECTION_MODE}" \
  --metric "${METRIC}" \
  --objective "${OBJECTIVE}" \
  --seed "${SEED}" \
  --train-frac "${TRAIN_FRAC}" \
  --plot-sweep-splits "${PLOT_SWEEP_SPLITS}"

echo "[OK] wrote defense outputs to ${OUT_DIR}"
