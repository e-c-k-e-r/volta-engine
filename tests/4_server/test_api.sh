#!/usr/bin/env bash
set -uo pipefail
# Starts volta_server and runs the black-box API checks (OpenAI chat completions, Anthropic messages, prefix cache).
# VOLTA_TEST_MODEL=glm5_next runs the same checks against GLM-5.3-Flash (no prefix cache there yet).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$DIR/tests/local.env" ] && . "$DIR/tests/local.env"
: "${ENGINE_MODEL_ROOT:?set ENGINE_MODEL_ROOT (or tests/local.env)}"
PORT="${VOLTA_TEST_PORT:-18080}"
LOG="$(mktemp -u /tmp/volta_api_test_XXXXXX.log)"

if [ "${VOLTA_TEST_MODEL:-qwen3_5}" = "glm5_next" ]; then
	MODEL_DIR="${ENGINE_TEST_GLM_DIR:-$ENGINE_MODEL_ROOT/GLM-5.3-Flash-exl3-3.05bpw}"
	DRAFT_DIR="${ENGINE_TEST_GLM_DRAFT_DIR:-$ENGINE_MODEL_ROOT/GLM-5.3-Flash-DFlash2}"
else
	MODEL_DIR="$ENGINE_MODEL_ROOT/Qwen3.8-27B-exl3"
	DRAFT_DIR="$ENGINE_MODEL_ROOT/Qwen3.8-27B-DFlash2-EXL3-5.0bpw"
fi
# VOLTA_TEST_SPEC=mtp speculates with the checkpoint's own MTP head instead of the DFlash2 drafter (Qwen)
if [ "${VOLTA_TEST_SPEC:-dflash}" = "mtp" ]; then
	SPEC_ARGS=(--speculative --set speculator=mtp)
else
	SPEC_ARGS=(--speculative --draft-model-dir "$DRAFT_DIR")
fi
taskset -c 0-9 "$DIR/build/volta_server" --port "$PORT" --host 127.0.0.1 --model-dir "$MODEL_DIR" \
	"${SPEC_ARGS[@]}" --set prefix_interval=1024 > "$LOG" 2>&1 &
PID=$!
cleanup() { kill -TERM "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 600); do
	curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
	kill -0 "$PID" 2>/dev/null || { echo "[-] server exited early (see $LOG)"; exit 1; }
	sleep 0.3
done
python3 -I "$DIR/tests/4_server/api_checks.py" "http://127.0.0.1:$PORT"
