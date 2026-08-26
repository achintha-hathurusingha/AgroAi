#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PIDFILE=serve_qwen3vl.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running (PID $(cat "$PIDFILE"))"
    exit 0
fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate agrivision
export HF_HOME=/home/minura/.cache/huggingface

nohup python3 serve_qwen3vl.py >> serve_qwen3vl.log 2>&1 &
echo $! > "$PIDFILE"
disown
echo "started, PID $(cat "$PIDFILE")"
