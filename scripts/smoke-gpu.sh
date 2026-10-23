#!/usr/bin/env bash
# Step 4c: run smoke_gpu.py in a throwaway container: GPU, no network, 8 GB memory cap.
#   bash ~/src/minimax-m31-flash-4x-spark/scripts/smoke-gpu.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG=vllm/vllm-openai:minimax-m31-flash-86f582da
docker run --rm --gpus all --network none --memory 8g \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v "$HERE/smoke_gpu.py":/smoke_gpu.py:ro \
  --entrypoint python3 "$TAG" /smoke_gpu.py
