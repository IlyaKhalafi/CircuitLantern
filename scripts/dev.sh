#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cleanup() {
  kill "${backend_pid:-}" "${web_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(cd backend && uv run uvicorn circuit_lantern.main:app --host 127.0.0.1 --port 8000) &
backend_pid=$!
(cd web && npm run dev) &
web_pid=$!
wait -n "$backend_pid" "$web_pid"
