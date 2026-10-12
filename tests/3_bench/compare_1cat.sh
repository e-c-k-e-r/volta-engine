#!/usr/bin/env bash
# Side-by-side runs with 1Cat-vLLM's published SM70 decode workloads (see bench_serve.py): starts a server for one arm,
# runs the benchmark against it, stops the server. Power cap is whatever the GPUs are set to now: it is recorded in the output.
#
#   tests/3_bench/compare_1cat.sh <arm> [bench_serve.py options...]
#
# arms:
#   qwen_dflash     Qwen3.8-27B EXL3 + DFlash2 (EXL3 5.0 bpw drafter)          their: 229 tok/s at 4K, 201 at 32K, 171 at 128K (NVFP4 + DFlash2 q7)
#   qwen_dflash_bf  the same with the BF16 drafter (as theirs)
#   qwen_nvfp4      Qwen3.8-27B QUASAR NVFP4 + BF16 DFlash2 drafter (their checkpoint pair)
#   qwen_plain      Qwen3.8-27B EXL3, no speculation (the sampled-step cost without acceptance effects)
#   qwen_mtp        Qwen3.8-27B EXL3 + its own MTP head
#   flash_mtp7      Flash-Next EXL3 4.0 bpw + NVFP4 experts / dense groups, MTP depth 7 (default)
#   flash_mtp4      the same at MTP depth 4 (their contract: 5-row verify)
#   flash_exl3      Flash-Next EXL3 4.0 bpw only, MTP depth 7
# defaults for the benchmark: --cases fixed4k,fixed8k,natural8k --repeats 3 (add --cases fixed32k, fixed128k, --concurrency 4 for the rest)
set -u
ARM=${1:?arm}; shift
M=${MODELS:-$HOME/Documents/models}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PORT=${PORT:-8300}
TAG=$(date +%Y%m%d_%H%M%S)
CAP=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
LOG=$ROOT/results/serve_${ARM}_cap${CAP}_$TAG
common=(--tp-size 4 --port $PORT --host 127.0.0.1 --set max_seq=262144 --set max_slots=2 --set prefix_cache_gib=0)
case $ARM in
	qwen_dflash)    args=(--model-dir $M/Qwen3.8-27B-exl3 --speculative --draft-model-dir $M/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) ;;
	qwen_dflash_bf) args=(--model-dir $M/Qwen3.8-27B-exl3 --speculative --draft-model-dir $M/Qwen3.8-27B-DFlash2) ;;
	qwen_nvfp4)     args=(--model-dir $M/Qwen3.8-27B-QUASAR-NVFP4 --speculative --draft-model-dir $M/Qwen3.8-27B-DFlash2) ;;
	qwen_plain)     args=(--model-dir $M/Qwen3.8-27B-exl3) ;;
	qwen_mtp)       args=(--model-dir $M/Qwen3.8-27B-exl3 --speculative --set speculator=mtp) ;;
	flash_mtp7|flash_mtp4|flash_exl3)
		depth=7; [ $ARM = flash_mtp4 ] && depth=4
		args=(--model-dir $M/Swift-Qwen3.8-Flash-Next-exl3-4.00bpw --speculative --set speculator=mtp --set mtp_depth=$depth)
		[ $ARM != flash_exl3 ] && args+=(--set nvfp4_experts=$M/Qwen3.8-Flash-Next-NVFP4 --set nvfp4_dense_dir=$M/Qwen3.8-Flash-Next-NVFP4-dense
			--set nvfp4_dense=gdn,attn,idx,head,shared_gate,shared_up) ;;
	*) echo "unknown arm $ARM"; exit 2 ;;
esac
mkdir -p "$ROOT/results"
echo "arm $ARM, power limit ${CAP} W, log $LOG.server.log"
cd "$ROOT"
setsid nohup taskset -c 0-9 build/volta_server "${args[@]}" "${common[@]}" > "$LOG.server.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
for i in $(seq 1 240); do
	curl -s -m 2 http://127.0.0.1:$PORT/health >/dev/null && break
	kill -0 $SP 2>/dev/null || { echo "server exited; see $LOG.server.log"; exit 1; }
	sleep 5
done
python3 tests/3_bench/bench_serve.py --url http://127.0.0.1:$PORT --out "$LOG.json" "$@" | tee "$LOG.txt"
