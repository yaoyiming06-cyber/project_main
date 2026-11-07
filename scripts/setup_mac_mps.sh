#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python 3.12 was not found. Install it with: brew install python@3.12" >&2
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import platform
import sys
if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit(f"Use Python 3.11 or 3.12; found {sys.version.split()[0]}")
if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
    raise SystemExit("This setup script is for Apple Silicon Macs")
PY

VENV="$PROJECT_ROOT/.venv-mac"
if [[ ! -x "$VENV/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements-mac.txt
"$VENV/bin/python" -m pip install -e "$PROJECT_ROOT" --no-deps
"$VENV/bin/python" scripts/check_environment.py --device mps

echo "Mac MPS environment ready. Next run scripts/download_weights.py and scripts/validate_data.py."
