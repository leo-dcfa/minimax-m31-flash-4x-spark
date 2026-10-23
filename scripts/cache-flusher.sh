#!/usr/bin/env bash
# Keep the page cache small on THIS node while the model loads: a GB10's GPU allocates from the same
# memory, and a full cache can leave the driver unable to place the KV cache (or freeze the node).
# Drops the cache whenever it holds more than LIMIT_GIB, for DURATION seconds; one log line per drop.
#   nohup bash cache-flusher.sh > /dev/null 2>&1 &        stop early: bash cache-flusher.sh --stop
# Needs passwordless sudo for exactly: tee /proc/sys/vm/drop_caches
set -u
DURATION="${DURATION:-2700}"; LIMIT_GIB="${LIMIT_GIB:-40}"
PIDFILE="$HOME/.m31f-cache-flusher.pid"; LOG="$HOME/models/m31f-cache-flusher.log"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then kill "$(cat "$PIDFILE")"; fi
[ "${1:-}" = --stop ] && exit 0
echo $$ > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
echo "$(date -Is) start: ${DURATION}s, drop when Cached > ${LIMIT_GIB} GiB" >> "$LOG"
end=$((SECONDS + DURATION)); n=0
while [ "$SECONDS" -lt "$end" ]; do
  cached=$(awk '/^Cached:/{print int($2/1048576)}' /proc/meminfo)
  if [ "${cached:-0}" -gt "$LIMIT_GIB" ]; then
    sync
    if echo 3 | sudo -n /usr/bin/tee /proc/sys/vm/drop_caches > /dev/null 2>&1; then
      n=$((n + 1))
      echo "$(date -Is) dropped: Cached was ${cached} GiB, MemAvailable now $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB" >> "$LOG"
    else
      echo "$(date -Is) drop_caches refused (no passwordless sudo for it)" >> "$LOG"; sleep 60
    fi
  fi
  sleep 5
done
echo "$(date -Is) done: $n drops" >> "$LOG"
