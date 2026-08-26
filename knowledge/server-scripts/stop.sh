#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PIDFILE=serve_qwen3vl.pid

if [[ ! -f "$PIDFILE" ]] || ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "not running"
    rm -f "$PIDFILE"
    exit 0
fi

PID=$(cat "$PIDFILE")
kill "$PID"
rm -f "$PIDFILE"
echo "stopped PID $PID (SIGTERM sent, frees VRAM on exit)"
