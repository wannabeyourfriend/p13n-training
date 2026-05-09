#!/usr/bin/env bash
# Watch GPUs; when one is idle (mem < THRESH_MIB, util < THRESH_UTIL),
# launch a queued training job pinned to that GPU.
# Survives terminal close via setsid + nohup. Use stop_auto_submit.sh to cancel.
#
# Usage:
#   ./scripts/auto_submit_train.sh <train_script.sh> [more_scripts.sh ...]
# Example:
#   ./scripts/auto_submit_train.sh scripts/train_qwen25_7b_instruct_rollout_oracle_profile_only.sh
set -euo pipefail
cd "$(dirname "$0")/.."

THRESH_MIB="${THRESH_MIB:-5000}"
THRESH_UTIL="${THRESH_UTIL:-10}"
POLL_SEC="${POLL_SEC:-60}"
STABLE_CYCLES="${STABLE_CYCLES:-3}"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <train_script.sh> [more_scripts.sh ...]" >&2
  exit 2
fi

QUEUE=("$@")
for s in "${QUEUE[@]}"; do
  [ -x "$s" ] || { echo "not executable: $s" >&2; exit 2; }
done

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
WATCH_LOG="$LOG_DIR/auto_submit_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/auto_submit.pid"

# Re-exec detached so closing the terminal does NOT kill the watcher.
if [ "${_DETACHED:-0}" != "1" ]; then
  export _DETACHED=1
  setsid nohup "$0" "$@" >"$WATCH_LOG" 2>&1 < /dev/null &
  disown || true
  echo $! > "$PID_FILE"
  echo "[auto-submit] detached pid=$(cat "$PID_FILE") log=$WATCH_LOG"
  echo "[auto-submit] queue:"
  printf '  - %s\n' "${QUEUE[@]}"
  exit 0
fi

log() { echo "[$(date '+%F %T')] $*"; }

declare -A STABLE
for i in 0 1 2 3; do STABLE[$i]=0; done
declare -A BUSY_GPU

log "watcher started. thresh_mib=$THRESH_MIB thresh_util=$THRESH_UTIL poll=${POLL_SEC}s stable=${STABLE_CYCLES}"
log "queue: ${QUEUE[*]}"

qidx=0
while [ "$qidx" -lt "${#QUEUE[@]}" ]; do
  mapfile -t ROWS < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  for row in "${ROWS[@]}"; do
    idx=$(echo "$row" | awk -F',' '{gsub(/ /,""); print $1}')
    mem=$(echo "$row" | awk -F',' '{gsub(/ /,""); print $2}')
    util=$(echo "$row" | awk -F',' '{gsub(/ /,""); print $3}')
    if [ -n "${BUSY_GPU[$idx]:-}" ]; then
      continue
    fi
    if [ "$mem" -lt "$THRESH_MIB" ] && [ "$util" -lt "$THRESH_UTIL" ]; then
      STABLE[$idx]=$((STABLE[$idx]+1))
      log "gpu=$idx idle (mem=${mem} util=${util}) stable=${STABLE[$idx]}/$STABLE_CYCLES"
      if [ "${STABLE[$idx]}" -ge "$STABLE_CYCLES" ] && [ "$qidx" -lt "${#QUEUE[@]}" ]; then
        script="${QUEUE[$qidx]}"
        stem="$(basename "$script" .sh)"
        run_log="$LOG_DIR/${stem}_gpu${idx}_$(date +%Y%m%d_%H%M%S).log"
        log "LAUNCH gpu=$idx script=$script log=$run_log"
        CUDA_VISIBLE_DEVICES="$idx" setsid nohup bash "$script" >"$run_log" 2>&1 < /dev/null &
        disown || true
        BUSY_GPU[$idx]=1
        qidx=$((qidx+1))
        STABLE[$idx]=0
        sleep 30
      fi
    else
      STABLE[$idx]=0
    fi
  done
  sleep "$POLL_SEC"
done

log "all ${#QUEUE[@]} job(s) launched. watcher exiting."
