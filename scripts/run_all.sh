#!/bin/bash
# Full pipeline: setup datasets → benchmark → plot results.
# Usage: bash scripts/run_all.sh [--dry-run] [--epochs N]

set -euo pipefail

PYTHON="${PYTHON:-python}"
EPOCHS=3
DRY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY="--dry-run" ;;
        --epochs)  EPOCHS="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "============================================================"
echo "Step 1: Generate datasets"
echo "============================================================"
bash scripts/setup_datasets.sh $DRY

echo ""
echo "============================================================"
echo "Step 2: Run benchmark (epochs=$EPOCHS)"
echo "============================================================"
$PYTHON experiments/run_benchmark.py --epochs "$EPOCHS" --repeats 3 $DRY

echo ""
echo "============================================================"
echo "Step 3: Plot results"
echo "============================================================"
$PYTHON experiments/plot_results.py

echo ""
echo "Pipeline complete. Results in results/disk_benchmark.csv"
echo "Figures in results/figures/"
