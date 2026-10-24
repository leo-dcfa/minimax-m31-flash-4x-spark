#!/usr/bin/env bash
# Step 6: boot the four ranks from the head (rank 0). Run inside tmux on the head:
#   tmux -L m31f new -d -s boot 'bash ~/src/minimax-m31-flash-4x-spark/scripts/launch.sh'
#   DRY_RUN=1 bash .../launch.sh          preflight + print each rank's docker command, start nothing
# 1. preflight every node: weights, image, the same patched NCCL, nothing else on the GPU, free memory
# 2. start the cache flushers, then ranks 3, 2, 1 (headless; they wait for the head), then rank 0
# 3. watch: stop all four ranks if any node falls below GUARD_GIB free, or a rank's container exits;
#    finish when the API answers /health
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
. "$HERE/site.env"
read -ra RING <<< "$RING_IPS"; read -ra PEERS <<< "$WORKERS_FROM_HEAD"
GUARD_GIB="${GUARD_GIB:-3}"; MIN_START_GIB="${MIN_START_GIB:-100}"; TIMEOUT_MIN="${TIMEOUT_MIN:-90}"
S="$HERE/scripts"
on() { local r=$1; shift; if [ "$r" = 0 ]; then bash -c "$*"; else ssh -n -o BatchMode=yes "${PEERS[$((r-1))]}" "$*"; fi; }
say() { echo "$(date +%T) $*"; }
stop_all() { for r in 3 2 1 0; do on "$r" "docker rm -f m31f-tp4 >/dev/null 2>&1; bash $S/cache-flusher.sh --stop"; done; }

say "1/3 preflight"
bad=0; nccl_sums=""
for r in 0 1 2 3; do
  out=$(on "$r" "
    n=\$(ls $MODEL_DIR/*.safetensors 2>/dev/null | wc -l)
    img=\$(docker image inspect -f ok vllm/vllm-openai:minimax-m31-flash-86f582da 2>/dev/null || echo missing)
    nccl=\$(sha256sum $NCCL_LIB | cut -c1-16)
    gpu=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    old=\$(docker ps -aq -f name=^m31f-tp4\$ | wc -l)
    swp=\$(cat /proc/sys/vm/swappiness)
    mem=\$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo)
    echo \"\$(hostname) shards=\$n image=\$img nccl=\$nccl gpu_procs=\$gpu old_container=\$old swappiness=\$swp free=\${mem}GiB\"")
  echo "  rank $r: $out"
  case "$out" in *" shards=48 image=ok "*" gpu_procs=0 old_container=0 swappiness=0 "*) ;; *) bad=1 ;; esac
  mem=${out##*free=}; mem=${mem%GiB}; [ "${mem:-0}" -ge "$MIN_START_GIB" ] 2>/dev/null || bad=1
  h=${out#*nccl=}; nccl_sums="$nccl_sums ${h%% *}"
done
# shellcheck disable=SC2086  # one hash per word
[ "$(printf '%s\n' $nccl_sums | sort -u | wc -l)" = 1 ] || { say "the four NCCL builds differ"; bad=1; }
[ "$bad" = 0 ] || { say "PREFLIGHT FAILED (needs: model, image, one NCCL build, no GPU process, swappiness 0, >= ${MIN_START_GIB} GiB free)"; exit 1; }
if [ -n "${DRY_RUN:-}" ]; then
  for r in 3 2 1 0; do echo "  rank $r: $(on "$r" "DRY_RUN=1 bash $S/run-rank.sh $r")"; done
  exit 0
fi

say "2/3 start"
if [ ! -f "$API_KEY_ENV" ]; then
  mkdir -p "$(dirname "$API_KEY_ENV")"
  (umask 077; printf 'VLLM_API_KEY=%s\n' "$(openssl rand -hex 24)" > "$API_KEY_ENV")
  say "made a new API key in $API_KEY_ENV (mode 600)"
fi
for r in 0 1 2 3; do on "$r" "nohup bash $S/cache-flusher.sh > /dev/null 2>&1 < /dev/null &"; done
for r in 3 2 1 0; do
  on "$r" "MOE_BACKEND='${MOE_BACKEND:-}' SPEC_CONFIG='${SPEC_CONFIG:-}' IMAGE_LIMIT='${IMAGE_LIMIT:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' bash $S/run-rank.sh $r" > /dev/null \
    || { say "rank $r did not start"; stop_all; exit 1; }
  say "  rank $r started"
done

say "3/3 watch (guard: stop everything below ${GUARD_GIB} GiB free on any node)"
start=$SECONDS; last=0
while [ $((SECONDS - start)) -lt $((TIMEOUT_MIN * 60)) ]; do
  line=""; low=0; dead=0
  for r in 0 1 2 3; do
    s=$(on "$r" "echo \$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo) \$(docker inspect -f '{{.State.Running}}' m31f-tp4 2>/dev/null)")
    [ -z "$s" ] && { line="$line r$r=unreachable"; continue; }   # one missed poll is not a failure
    m=${s%% *}; run=${s##* }
    line="$line r$r=${m}GiB"; [ "$m" -lt "$GUARD_GIB" ] && low=1; [ "$run" = true ] || dead=1
  done
  [ "$low" = 1 ] && { say "GUARD:$line, stopping all ranks"; stop_all; exit 2; }
  [ "$dead" = 1 ] && { say "a rank's container exited:$line; see docker logs m31f-tp4 on each node"; stop_all; exit 3; }
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://$API_HOST:$API_PORT/health" || true)
  [ "$code" = 200 ] && { say "API up after $(( (SECONDS - start) / 60 )) min:$line"; exit 0; }
  [ $((SECONDS - last)) -ge 60 ] && { say "loading $(( (SECONDS - start) / 60 )) min:$line"; last=$SECONDS; }
  sleep 10
done
say "no API after ${TIMEOUT_MIN} min, stopping"; stop_all; exit 4
