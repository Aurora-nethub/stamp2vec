#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  echo "Missing .venv. Please create it first."
  exit 1
fi

source "${ROOT_DIR}/.venv/bin/activate"
python "${ROOT_DIR}/main.py"
