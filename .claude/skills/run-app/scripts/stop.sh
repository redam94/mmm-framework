#!/usr/bin/env bash
# Stop the dev servers by PORT. `pkill -f uvicorn` misses the backend because
# `uv run uvicorn` shows up as `python3.1`, so kill whatever is listening.
set -u
for port in 8000 5173; do
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    kill -9 $pids 2>/dev/null || true
    echo "stopped :$port (pids: $pids)"
  else
    echo ":$port already free"
  fi
done
