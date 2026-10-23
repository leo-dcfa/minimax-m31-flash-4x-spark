#!/usr/bin/env bash
# Step 3: copy the verified weights between nodes over the CX7 cables (rsync over ssh).
#   copy.sh --from <host>             pull MODEL_DIR from <host> to this node
#   copy.sh --to <host> [<host> ...]  push this node's MODEL_DIR to each host, in parallel
# Give each host by its address on a CX7 cable or on ring0, so the bytes cross the fabric and
# not the LAN. Same MODEL_DIR path on every node. Run verify.sh on the receiving node afterwards.
#
# Memory: every rsync is capped on BOTH ends. Locally through systemd-run (all the pushes of one
# --to share one cap), remotely through --rsync-path, which starts the far rsync inside its own
# capped scope. Both caps hold each side's page cache too, which on a GB10 is GPU memory.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$HERE/site.env" ] && . "$HERE/site.env"
DEST="${MODEL_DIR:-$HOME/models/MiniMax-M3.1-Flash}"
CAP=(systemd-run --user --scope --quiet -p MemoryHigh=3G -p MemoryMax=5G)

# AES-GCM is hardware-accelerated on Grace; compression only costs CPU on safetensors.
SSH="ssh -T -o Compression=no -o BatchMode=yes -c aes128-gcm@openssh.com"
REMOTE_RSYNC="mkdir -p '$DEST' && ${CAP[*]} rsync"
OPTS=(-a --partial --no-inc-recursive --exclude .cache/ --info=stats1,progress2
      -e "$SSH" --rsync-path="$REMOTE_RSYNC")

push_all() {
  local pids=() rc=0 h
  for h in "$@"; do
    rsync "${OPTS[@]}" "$DEST/" "$h:$DEST/" > "$DEST.copy-to-$h.log" 2>&1 &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return "$rc"
}

case "${1:-}" in
  --from)
    mkdir -p "$DEST"
    "${CAP[@]}" rsync "${OPTS[@]}" "$2:$DEST/" "$DEST/" 2>&1 | tee "$DEST.copy-from.log"
    exit "${PIPESTATUS[0]}"
    ;;
  --to)
    shift
    "${CAP[@]}" bash "$0" --push-all "$@"
    ;;
  --push-all)
    shift
    push_all "$@"
    ;;
  *)
    sed -n '2,5p' "$0"
    exit 2
    ;;
esac
