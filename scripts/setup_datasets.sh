#!/bin/bash
# Generate synthetic ImageNet-style datasets on all 3 storage locations.
# Usage: bash scripts/setup_datasets.sh [--dry-run]

set -euo pipefail

PYTHON="${PYTHON:-python}"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

for ROOT in "/home/cake11298/datasets" "/mnt/ssd/datasets" "/mnt/hdd/datasets"; do
    echo "=== Generating dataset at $ROOT ==="
    $PYTHON src/dataset_gen.py --root "$ROOT" --classes 20 --images-per-class 200 $DRY
done

echo ""
echo "Dataset generation complete."
