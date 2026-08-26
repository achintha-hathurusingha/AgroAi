#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PIDFILE=serve_qwen3vl.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "running, PID $(cat "$PIDFILE")"
else
    echo "not running"
fi
echo "---- last 15 log lines ----"
tail -n 15 serve_qwen3vl.log 2>/dev/null || echo "(no log yet)"
