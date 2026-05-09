#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PID_FILE="outputs/logs/auto_submit.pid"
if [ ! -f "$PID_FILE" ]; then
  echo "no pid file: $PID_FILE"
  exit 0
fi
pid=$(cat "$PID_FILE")
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "killed auto-submit pid=$pid"
else
  echo "pid=$pid not running"
fi
rm -f "$PID_FILE"
