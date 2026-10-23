#!/usr/bin/env bash
# Step 4a: pull the vLLM image on THIS node by digest, tag it readably, then drop the page cache.
#   bash ~/src/minimax-m31-flash-4x-spark/scripts/pull-image.sh
# Pull on each node directly (not docker save | docker load): only a pull records the digest.
# Unpacking ~11.5 GB of layers writes through the page cache, which on a GB10 is GPU memory too,
# so the cache is dropped afterwards (needs passwordless sudo for exactly that tee; skipped if not).
set -euo pipefail
IMAGE=vllm/vllm-openai@sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c
TAG=vllm/vllm-openai:minimax-m31-flash-86f582da
MIN_FREE_GIB="${MIN_FREE_GIB:-10}"

free_gib() { free -g | awk '/Mem/{print $7}'; }
[ "$(free_gib)" -ge "$MIN_FREE_GIB" ] || { echo "only $(free_gib) GiB free, need $MIN_FREE_GIB: not pulling"; exit 3; }

echo "$(hostname): pulling $IMAGE ($(free_gib) GiB free)"
start=$(date +%s)
docker pull -q "$IMAGE"
docker tag "$IMAGE" "$TAG"
echo "pulled in $(( $(date +%s) - start )) s; $(docker image inspect "$TAG" --format '{{.Architecture}} {{.Os}} {{.Size}} bytes')"
docker image inspect "$TAG" --format 'repo digests: {{join .RepoDigests " "}}'
sync
if echo 3 | sudo -n /usr/bin/tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
  echo "page cache dropped; $(free_gib) GiB free"
else
  echo "page cache NOT dropped (no passwordless sudo for it); $(free_gib) GiB free"
fi
