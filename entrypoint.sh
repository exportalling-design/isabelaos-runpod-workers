#!/bin/bash
set -e

echo "[ENTRYPOINT] starting worker..."
exec python3 -u /app/worker.py
