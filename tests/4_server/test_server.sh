#!/usr/bin/env bash
set -euo pipefail

# Phase 4 Server Integration Verification Suite for Volta SM70 (4x V100-SXM2-32GB)
echo "========================================================"
echo "   VOLTA ENGINE: INTEGRATION TEST SUITE (PHASE 4)       "
echo "========================================================"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVER_BIN="$DIR/build/volta_server"
SERVER_LOG="/tmp/volta_server_test.log"
PORT=8000
HOST="127.0.0.1"
MODEL_DIR="/home/mrq/Documents/models/Qwen3.8-27B-exl3"

DRAFT_DIR="/home/mrq/Documents/models/Qwen3.8-27B-DFlash2-EXL3-5.0bpw"

if [ ! -f "$SERVER_BIN" ]; then
	echo "[-] FATAL: Server binary $SERVER_BIN not found! Build it first."
	exit 1
fi

rm -f "$SERVER_LOG"

echo "[*] Launching volta_server detached under taskset -c 0-9 (speculative enabled)..."
taskset -c 0-9 "$SERVER_BIN" --port "$PORT" --host "$HOST" --load-weights --model-dir "$MODEL_DIR" --use-speculation --draft-model-dir "$DRAFT_DIR" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
	if kill -0 "$SERVER_PID" 2>/dev/null; then
		echo "[*] Cleanup: Sending SIGTERM to volta_server (PID: $SERVER_PID)..."
		kill -TERM "$SERVER_PID" 2>/dev/null || true
		for t in $(seq 1 30); do
			if ! kill -0 "$SERVER_PID" 2>/dev/null; then
				break
			fi
			sleep 0.1
		done
		if kill -0 "$SERVER_PID" 2>/dev/null; then
			echo "[*] Cleanup: Sending SIGKILL to volta_server (PID: $SERVER_PID)..."
			kill -KILL "$SERVER_PID" 2>/dev/null || true
		fi
	fi
}
trap cleanup EXIT INT TERM

echo "[*] Polling http://$HOST:$PORT/health..."
READY=0
for i in $(seq 1 50); do
	if curl -s -f "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":\s*"ok"'; then
		READY=1
		echo "[+] Server is ready! Health check passed."
		break
	fi
	sleep 0.3
done

if [ "$READY" -ne 1 ]; then
	echo "[-] FATAL: Server failed to respond to /health within 15 seconds."
	if [ -f "$SERVER_LOG" ]; then
		echo "--- Server Log Excerpt ---"
		tail -n 30 "$SERVER_LOG"
	fi
	exit 1
fi

# 1. Test Health endpoint
echo "[*] STEP 1: Verifying GET /health..."
HEALTH_RESP=$(curl -s "http://$HOST:$PORT/health")
echo "    Response: $HEALTH_RESP"
if ! echo "$HEALTH_RESP" | grep -q '"status":\s*"ok"'; then
	echo "[-] FATAL: Invalid health response!"
	exit 1
fi
echo "[+] Health check OK."

# 2. Test Non-Streaming Chat Completion
echo "[*] STEP 2: Verifying non-streaming POST /v1/chat/completions..."
COMP_RESP=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
	-H "Content-Type: application/json" \
	-d '{"model":"qwen38","messages":[{"role":"user","content":"What is the capital of France?"}],"stream":false,"max_tokens":10}')
echo "    Response: $COMP_RESP"
if ! echo "$COMP_RESP" | grep -q "Paris"; then
	echo "[-] FATAL: Non-streaming completion did not contain expected answer Paris!"
	exit 1
fi
echo "[+] Non-streaming completion OK."

# 3. Test Streaming SSE Chat Completion
echo "[*] STEP 3: Verifying streaming SSE POST /v1/chat/completions..."
SSE_RESP=$(curl -s -N -X POST "http://$HOST:$PORT/v1/chat/completions" \
	-H "Content-Type: application/json" \
	-d '{"model":"qwen38","messages":[{"role":"user","content":"List the first 5 prime numbers:"}],"stream":true,"max_tokens":20}')

echo "    SSE Output sample:"
echo "$SSE_RESP"
if ! echo "$SSE_RESP" | grep -q "data: \[DONE\]"; then
	echo "[-] FATAL: Streaming SSE completion missing data: [DONE] termination marker!"
	exit 1
fi
echo "[+] Streaming SSE completion OK."

# 3b. Single-stream speculative throughput (Case A: one request owns the drafter context)
echo "[*] STEP 3b: Single-stream speculative throughput..."
SS_TOKENS=200
SS_GATE=90
SS_START=$(date +%s%N)
SS_RESP=$(curl -s -N -X POST "http://$HOST:$PORT/v1/chat/completions" \
	-H "Content-Type: application/json" \
	-d '{"model":"qwen38","messages":[{"role":"user","content":"Explain how a hash table handles collisions, and compare chaining with open addressing."}],"stream":true,"max_tokens":'"$SS_TOKENS"'}')
SS_END=$(date +%s%N)
SS_CHUNKS=$(echo "$SS_RESP" | grep -c 'content' || true)
SS_SEC=$(awk -v a="$SS_START" -v b="$SS_END" 'BEGIN { printf "%.4f", (b - a) / 1000000000 }')
SS_TPS=$(awk -v t="$SS_CHUNKS" -v s="$SS_SEC" 'BEGIN { printf "%.1f", t / s }')
echo "    $SS_CHUNKS chunks in ${SS_SEC} s: ${SS_TPS} tok/s single stream (greedy M=1 is ~60 tok/s; gate >= $SS_GATE)"
SS_INT=${SS_TPS%.*}
if [ "$SS_INT" -lt "$SS_GATE" ]; then
	echo "[-] FATAL: single-stream speculative throughput ${SS_TPS} tok/s below ${SS_GATE} tok/s!"
	exit 1
fi
echo "[+] Single-stream speculative throughput OK."

# 4. Concurrency Gate: 8 Simultaneous Client Streams
echo "[*] STEP 4: Concurrency Gate (8 Simultaneous Streams)..."
CONCURRENCY=8
TOKENS_PER_CLIENT=50
EXPECTED_TOTAL_TOKENS=$(( CONCURRENCY * TOKENS_PER_CLIENT ))

TEMP_DIR=$(mktemp -d /tmp/volta_stream_XXXXXX)
START_TIME=$(date +%s%N)

CLIENT_PIDS=()
for i in $(seq 1 $CONCURRENCY); do
	curl -s -N -X POST "http://$HOST:$PORT/v1/chat/completions" \
		-H "Content-Type: application/json" \
		-d '{"model":"qwen38","messages":[{"role":"user","content":"Tell me about distributed computing on Volta GPUs."}],"stream":true,"max_tokens":'"$TOKENS_PER_CLIENT"'}' \
		> "$TEMP_DIR/stream_$i.txt" &
	CLIENT_PIDS+=($!)
done

for pid in "${CLIENT_PIDS[@]}"; do
	wait "$pid"
done

END_TIME=$(date +%s%N)
ELAPSED_NS=$(( END_TIME - START_TIME ))
ELAPSED_SEC=$(awk -v ns="$ELAPSED_NS" 'BEGIN { printf "%.4f", ns / 1000000000 }')

TOTAL_TOKENS=0
for i in $(seq 1 $CONCURRENCY); do
	# Count chunks with delta content
	CHUNKS=$(grep -c 'content' "$TEMP_DIR/stream_$i.txt" || true)
	TOTAL_TOKENS=$(( TOTAL_TOKENS + CHUNKS ))
done
rm -rf "$TEMP_DIR"

THROUGHPUT=$(awk -v tok="$TOTAL_TOKENS" -v s="$ELAPSED_SEC" 'BEGIN { printf "%.1f", tok / s }')
echo "[+] 8 Concurrent Streams Results:"
echo "    Total Tokens Received: $TOTAL_TOKENS (Target: ~$EXPECTED_TOTAL_TOKENS)"
echo "    Elapsed Time: ${ELAPSED_SEC} s"
echo "    Aggregate Throughput: ${THROUGHPUT} tok/s (Target >= 150.0 tok/s)"

THROUGHPUT_INT=${THROUGHPUT%.*}
if [ "$THROUGHPUT_INT" -lt 150 ]; then
	echo "[-] FATAL: Aggregate throughput $THROUGHPUT tok/s below 150.0 tok/s exit gate!"
	exit 1
fi
echo "[+] Concurrency Exit Gate PASS (${THROUGHPUT} tok/s >= 150.0 tok/s)."

# 5. Graceful SIGTERM Shutdown & VRAM Teardown Audit
echo "[*] STEP 5: Testing Graceful Server Shutdown (SIGTERM)..."
kill -TERM "$SERVER_PID"
for t in $(seq 1 30); do
	if ! kill -0 "$SERVER_PID" 2>/dev/null; then
		break
	fi
	sleep 0.1
done
if kill -0 "$SERVER_PID" 2>/dev/null; then
	kill -KILL "$SERVER_PID" 2>/dev/null || true
fi
trap - EXIT INT TERM
sleep 1

echo "[*] STEP 6: Auditing GPU VRAM Cleanup across all 4 GPUs..."
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
echo "[+] VRAM in use across 4x V100: ${VRAM_USED} MiB"

if [ "$VRAM_USED" -ne 0 ]; then
	echo "[-] FATAL: VRAM leak detected! Total VRAM used: ${VRAM_USED} MiB (expected 0 MiB)"
	nvidia-smi
	exit 1
fi
echo "[+] 0 MiB VRAM confirmed across all 4 GPUs."

echo "========================================================"
echo "   [+] ALL PHASE 4 INTEGRATION TESTS PASSED!            "
echo "========================================================"
