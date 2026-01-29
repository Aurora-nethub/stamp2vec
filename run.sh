#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "${ROOT_DIR}/.venv" ]; then
  echo "Missing .venv. Please create it first."
  exit 1
fi

source "${ROOT_DIR}/.venv/bin/activate"

echo "Initializing Milvus database..."
python "${ROOT_DIR}/scripts/init_milvus.py"

if [ $? -ne 0 ]; then
  echo "Milvus initialization failed"
  exit 1
fi

echo "Starting Seal Embedding API..."
python "${ROOT_DIR}/main.py"
