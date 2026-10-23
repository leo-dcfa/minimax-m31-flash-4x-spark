#!/usr/bin/env bash
# Step 1: download MiniMax-M3.1-Flash at a pinned revision into MODEL_DIR on THIS node.
#
# Run it on a worker node (not the head, which also runs the API server), inside tmux so it
# survives the ssh session:
#   tmux -L m31f new -d -s download 'bash ~/src/minimax-m31-flash-4x-spark/scripts/download.sh'
# Watch:   tmux -L m31f attach -t download        (detach: Ctrl-b d; the session ends with the download)
#          bash ~/src/minimax-m31-flash-4x-spark/scripts/download-status.sh
# Stop:    pkill -TERM -f 'hf download MiniMaxAI/MiniMax-M3.1-Flash'
# Resume:  run it again; hf skips finished files and continues partial ones.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$HERE/site.env" ] && . "$HERE/site.env"

REPO=MiniMaxAI/MiniMax-M3.1-Flash
# main on 2026-10-23 02:26 UTC. The 02:23 commit (f6b8bc8f) differs only in README.md.
REV="${REV:-76faee1b4ac0f412e547781172e94548a6526a88}"
DEST="${MODEL_DIR:-$HOME/models/MiniMax-M3.1-Flash}"
HF="${HF:-hf}"
LOG="$DEST.download.log"

mkdir -p "$DEST"
echo "$(date -Is) start $REPO@$REV -> $DEST" | tee -a "$LOG"

# Why systemd-run: a GB10 has ONE pool of memory for the CPU and the GPU. A 252 GB download
# fills the page cache as it writes, and a node that is also serving a model may have only
# ~15 GiB free. The scope caps what this download may hold (its own memory plus the page cache
# of the files it writes): above MemoryHigh the kernel reclaims this scope's cache first, and at
# MemoryMax only the download is killed, never what else runs on the node.
#
# Which files: the weights (*.safetensors, including dspark/), configs and tokenizer (*.json,
# *.txt), the chat template (*.jinja), the licence and the model card. NOT the *.py files:
# they are MiniMax's remote code for transformers; vLLM has its own config and processor
# classes for this model, and this recipe never runs --trust-remote-code.
#
# HF_XET_HIGH_PERFORMANCE=0 keeps xet's parallel range downloads modest; the chunk cache is
# off because nothing re-reads these files from xet.
set +e
systemd-run --user --scope --quiet -p MemoryHigh=5G -p MemoryMax=7G \
  env HF_HUB_DISABLE_TELEMETRY=1 HF_XET_HIGH_PERFORMANCE=0 HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  "$HF" download "$REPO" --revision "$REV" --local-dir "$DEST" --max-workers 4 \
    --include '*.safetensors' --include '*.json' --include '*.jinja' --include '*.txt' \
    --include 'LICENSE' --include 'README*.md' --include 'docs/*.md' \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "$(date -Is) exit $rc" | tee -a "$LOG"
echo "$rc" > "$DEST.download.exit"
exit "$rc"
