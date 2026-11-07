#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-$PROJECT_ROOT/../cxr_first}"
PYTHON_PATH="$PROJECT_ROOT/.venv-mac/bin/python"

if [[ ! -x "$PYTHON_PATH" ]]; then
    echo "Mac environment is missing. Run scripts/setup_mac_mps.sh first." >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Could not find the image tree at $DATA_ROOT. Pass its path as the first argument." >&2
    exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_PATH" -m cxr_project.train \
    --config configs/g1_seed42_mac.yaml \
    --data-root "$DATA_ROOT" \
    --device mps
