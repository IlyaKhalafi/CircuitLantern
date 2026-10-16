#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

backend_port="${LANTERN_TEST_BACKEND_PORT:-8011}"
web_port="${LANTERN_TEST_WEB_PORT:-3110}"
run_dir="$(mktemp -d -t lantern-ui.XXXXXX)"
backend_pid=""
web_pid=""
web_snapshot="$run_dir/web"

cleanup() {
  if [[ -n "$web_pid" ]]; then kill "$web_pid" 2>/dev/null || true; wait "$web_pid" 2>/dev/null || true; fi
  if [[ -n "$backend_pid" ]]; then kill "$backend_pid" 2>/dev/null || true; wait "$backend_pid" 2>/dev/null || true; fi
  rm -rf "$run_dir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_service() {
  local target_url="$1"
  local log_file="$2"
  local service_pid="$3"
  for attempt in $(seq 1 120); do
    if ! kill -0 "$service_pid" 2>/dev/null; then
      cat "$log_file"
      return 1
    fi
    if curl --fail --silent --max-time 1 "$target_url" > /dev/null; then return 0; fi
    sleep 1
  done
  cat "$log_file"
  return 1
}

if [[ ! -x backend/.venv/bin/python || ! -d web/node_modules/next ]]; then
  echo "Install the project with make install before running browser tests."
  exit 1
fi

# Refuse to reuse an existing service so tests cannot mutate a research session.
backend/.venv/bin/python -c '
import socket, sys
for port in map(int, sys.argv[1:]):
    with socket.socket() as connection:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            connection.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(f"Test port {port} is in use. Choose different LANTERN_TEST_BACKEND_PORT and LANTERN_TEST_WEB_PORT values.")
' "$backend_port" "$web_port"

# Build an independent snapshot so tests cannot replace a running Next.js build
# or rewrite generated types in the working tree. Read only dependency files use
# hard links on the same filesystem, preserving package links without another
# installation. Source files and generated build output remain independent.
backend/.venv/bin/python - "$web_snapshot" <<'PY'
import os
import shutil
import sys
from pathlib import Path

def copy_dependency(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)

shutil.copytree(
    "web",
    sys.argv[1],
    symlinks=True,
    ignore=shutil.ignore_patterns(
        ".next", "node_modules", "test-results", "*.tsbuildinfo", ".env", ".env.*", ".git", "AGENTS.md"
    ),
)
shutil.copytree(
    "web/node_modules",
    Path(sys.argv[1]) / "node_modules",
    symlinks=True,
    copy_function=copy_dependency,
)
PY

(
  cd backend
  exec .venv/bin/python tests/serve_ui.py --port "$backend_port"
) > "$run_dir/backend.log" 2>&1 &
backend_pid=$!
wait_for_service "http://127.0.0.1:$backend_port/api/health" "$run_dir/backend.log" "$backend_pid"

(
  cd "$web_snapshot"
  npm run build
  LANTERN_BACKEND_URL="http://127.0.0.1:$backend_port" exec node node_modules/next/dist/bin/next start --hostname 127.0.0.1 --port "$web_port"
) > "$run_dir/web.log" 2>&1 &
web_pid=$!
wait_for_service "http://127.0.0.1:$web_port" "$run_dir/web.log" "$web_pid"

(
  cd web
  LANTERN_TEST_URL="http://127.0.0.1:$web_port" npm run test:e2e -- "$@"
)
