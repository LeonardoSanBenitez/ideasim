#!/usr/bin/env bash
# Run evaluation of the ideology simulation against a dataset.
#
# Usage:
#   ./scripts/evaluate.sh                          # single split, default dataset
#   ./scripts/evaluate.sh --n-folds 3              # 3-fold temporal CV
#   ./scripts/evaluate.sh path/to/dataset.json     # custom dataset
#   ./scripts/evaluate.sh --help                   # show all options
#
# Results are saved to output/evaluation/ by default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default dataset path
DEFAULT_DATASET="data/datasets/european_rough_estimates.json"

# Parse arguments: if first arg is a file path (not a flag), use it as dataset
DATASET=""
EXTRA_ARGS=()

for arg in "$@"; do
    if [[ -z "$DATASET" && ! "$arg" == --* && -f "$arg" ]]; then
        DATASET="$arg"
    else
        EXTRA_ARGS+=("$arg")
    fi
done

if [[ -z "$DATASET" ]]; then
    DATASET="$DEFAULT_DATASET"
fi

echo "=== IdeaSim Evaluation ==="
echo "Dataset: $DATASET"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

python3 -m ideasim evaluate "$DATASET" "${EXTRA_ARGS[@]}"
