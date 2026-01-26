#!/usr/bin/env bash
set -e

echo "[entrypoint] Starting worker..."

# Si tu volume trae venv, úsalo:
if [ -x "/workspace/wan22_env/bin/python" ]; then
  echo "[entrypoint] Using venv python: /workspace/wan22_env/bin/python"
  /workspace/wan22_env/bin/python /app/worker.py
else
  echo "[entrypoint] Using container python: python3"
  python3 /app/worker.py
fi
