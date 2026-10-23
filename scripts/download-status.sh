#!/usr/bin/env bash
# How far the download is: shards finished, bytes on disk, the node's free memory, the exit code.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$HERE/site.env" ] && . "$HERE/site.env"
DEST="${MODEL_DIR:-$HOME/models/MiniMax-M3.1-Flash}"
TOTAL_GB=252.4
done_shards=$(find "$DEST" -name '*.safetensors' -not -path '*/.cache/*' 2>/dev/null | wc -l)
bytes=$(du -sb "$DEST" 2>/dev/null | cut -f1)
gb=$(awk -v b="${bytes:-0}" 'BEGIN{printf "%.1f", b/1e9}')
echo "shards done: $done_shards of 49   on disk: $gb of $TOTAL_GB GB (partials included)"
free -g | awk '/Mem/{print "node MemAvailable: " $7 " GiB"}'
if [ -f "$DEST.download.exit" ]; then
  echo "finished, exit code $(cat "$DEST.download.exit")"
else
  pgrep -af 'hf download MiniMaxAI/MiniMax-M3.1-Flash' >/dev/null && echo "running" || echo "not running"
fi
