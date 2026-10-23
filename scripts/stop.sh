#!/usr/bin/env bash
# Stop the recipe on all four nodes, from the head: remove m31f-tp4 and stop the cache flushers.
# The weights, the image and the m31f-cache volume stay.
#   bash ~/src/minimax-m31-flash-4x-spark/scripts/stop.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
. "$HERE/site.env"
read -ra PEERS <<< "$WORKERS_FROM_HEAD"
S="$HERE/scripts"
for r in 3 2 1 0; do
  c="docker rm -f m31f-tp4 >/dev/null 2>&1 && echo removed || echo 'not running'; bash $S/cache-flusher.sh --stop"
  if [ "$r" = 0 ]; then out=$(bash -c "$c"); else out=$(ssh -n -o BatchMode=yes "${PEERS[$((r-1))]}" "$c"); fi
  echo "rank $r: $out"
done
