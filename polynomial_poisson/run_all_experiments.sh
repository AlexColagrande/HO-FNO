#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
REGENERATE="${REGENERATE:-0}"
GENERATED_DIR="$ROOT_DIR/generated"
RUNS_DIR="$ROOT_DIR/runs"

for degree in 1 2 3 5; do
  dataset="$GENERATED_DIR/polynomial_source_poisson_p${degree}/dataset.pt"
  if [[ "$REGENERATE" == "1" || ! -f "$dataset" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/create_dataset_polynomial_source_poisson_p2.py" \
      --output-root "$GENERATED_DIR" \
      --degree "$degree"
  fi
done

for dataset_name in \
  polynomial_source_poisson_p1 \
  polynomial_source_poisson_p2 \
  polynomial_source_poisson_p3 \
  polynomial_source_poisson_p5; do
  "$PYTHON_BIN" "$ROOT_DIR/train_and_eval_models.py" \
    --dataset "$GENERATED_DIR/$dataset_name/dataset.pt" \
    --output-root "$RUNS_DIR" \
    --epochs "$EPOCHS" \
    --batch-size 20 \
    --lr 0.001 \
    --device "$DEVICE"
done
