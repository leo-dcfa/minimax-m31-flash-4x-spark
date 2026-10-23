#!/usr/bin/env bash
# Step 2 (and again after step 3 on each node): verify the weights on THIS node against the
# manifest. Reads every byte once; changes nothing.
#   bash ~/src/minimax-m31-flash-4x-spark/scripts/verify.sh
# Output also goes to <MODEL_DIR>.verify.log.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$HERE/site.env" ] && . "$HERE/site.env"
DEST="${MODEL_DIR:-$HOME/models/MiniMax-M3.1-Flash}"
MANIFEST="${MANIFEST:-$HERE/manifest/76faee1b.json}"
WORKERS="${WORKERS:-4}"

# Same memory cap as the download: reading 252 GB fills the page cache, and on a GB10 that
# memory is shared with the GPU. verify.py also drops each file from the cache after hashing it.
systemd-run --user --scope --quiet -p MemoryHigh=4G -p MemoryMax=6G \
  python3 -I "$HERE/scripts/verify.py" "$MANIFEST" "$DEST" "$WORKERS" 2>&1 | tee "$DEST.verify.log"
exit "${PIPESTATUS[0]}"
