#!/usr/bin/env bash
# Start rank R (0-3) of the recipe on THIS node as the detached container m31f-tp4.
# launch.sh calls it on every node; this file is the recipe's command in one place.
#   bash run-rank.sh <R>              DRY_RUN=1 prints the docker command instead
# Knobs for later steps, one change per boot: MOE_BACKEND (default marlin), SPEC_CONFIG (DSpark
# JSON, step 8), IMAGE_LIMIT (images a prompt may carry, default 0: see README, Known issues),
# EXTRA_ARGS (plain flags such as --enforce-eager).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
. "$HERE/site.env"
R="${1:?usage: run-rank.sh <rank 0-3>}"
read -ra RING <<< "$RING_IPS"
TAG=vllm/vllm-openai:minimax-m31-flash-86f582da
NCCL_IN_IMAGE=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
VLLM_IN_IMAGE=/usr/local/lib/python3.12/dist-packages/vllm

cmd=(docker run -d --name m31f-tp4
  --gpus all --network host --shm-size 16g
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1
  -v "$MODEL_DIR":/model:ro
  -v "$NCCL_LIB":"$NCCL_IN_IMAGE":ro
  -v "$HERE/patches/minimax_m3_reasoning_parser.py":"$VLLM_IN_IMAGE/reasoning/minimax_m3_reasoning_parser.py":ro
  -v m31f-cache:/root/.cache
  -e TRITON_CACHE_DIR=/root/.cache/triton
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1
  -e VLLM_HOST_IP="${RING[$R]}"
  -e VLLM_WORKER_MULTIPROC_METHOD=fork -e VLLM_ENGINE_READY_TIMEOUT_S=5400
  --env-file "$HERE/nccl.env")
[ "$R" = 0 ] && cmd+=(--env-file "$API_KEY_ENV")
cmd+=("$TAG" /model --served-model-name minimax-m3.1-flash
  --tensor-parallel-size 4 --distributed-executor-backend mp
  --nnodes 4 --node-rank "$R" --master-addr "${RING[0]}" --master-port 29511
  --block-size 128 --disable-hybrid-kv-cache-manager
  --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype": "fp8"}'
  --gpu-memory-utilization 0.80 --kv-cache-memory-bytes 21474836480
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192
  --moe-backend "${MOE_BACKEND:-marlin}"
  --enable-prefix-caching --mm-encoder-tp-mode data --skip-mm-profiling
  --limit-mm-per-prompt "{\"image\": ${IMAGE_LIMIT:-0}, \"video\": 0}"
  --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice)
if [ "$R" = 0 ]; then cmd+=(--host "$API_HOST" --port "$API_PORT"); else cmd+=(--headless); fi
[ -n "${SPEC_CONFIG:-}" ] && cmd+=(--speculative-config "$SPEC_CONFIG")
# shellcheck disable=SC2206  # EXTRA_ARGS is a list of plain flags
[ -n "${EXTRA_ARGS:-}" ] && cmd+=($EXTRA_ARGS)

if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${cmd[@]}"; echo; exit 0; fi
"${cmd[@]}"
