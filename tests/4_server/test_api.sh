#!/usr/bin/env bash
set -uo pipefail
# Starts volta_server on the Qwen checkpoint and runs the black-box API checks (OpenAI chat completions).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$DIR/tests/local.env" ] && . "$DIR/tests/local.env"
: "${VOLTA_MODEL_ROOT:?set VOLTA_MODEL_ROOT (or tests/local.env)}"
PORT="${VOLTA_TEST_PORT:-18080}"
LOG="$(mktemp -u /tmp/volta_api_test_XXXXXX.log)"

taskset -c 0-9 "$DIR/build/volta_server" --port "$PORT" --host 127.0.0.1 --model-dir "$VOLTA_MODEL_ROOT/Qwen3.8-27B-exl3" \
	--speculative --draft-model-dir "$VOLTA_MODEL_ROOT/Qwen3.8-27B-DFlash2-EXL3-5.0bpw" > "$LOG" 2>&1 &
PID=$!
cleanup() { kill -TERM "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 600); do
	curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
	kill -0 "$PID" 2>/dev/null || { echo "[-] server exited early (see $LOG)"; exit 1; }
	sleep 0.3
done
python3 -I "$DIR/tests/4_server/api_checks.py" "http://127.0.0.1:$PORT"
