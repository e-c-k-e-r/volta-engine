#!/bin/zsh
# Regression gate for refactors. Modes (GPU modes serialize: one job at a time, run under taskset -c 0-9):
#   quick           CUDA build + kernel resource fingerprint (cuobjdump -res-usage: seconds) + coherence. For moves/renames/config work.
#   fingerprint     save the resource fingerprint of the current build (results/fingerprint.txt)
#   sass            full SASS md5 + HMMA count of bench_decode and volta_server in one pass (~40 s per binary), vs results/sass.txt (saved if absent)
#   static          no GPU: header hygiene greps (backend abstraction) and a full build
#   parity          GPU: tier 2 parity tests + coherence + sampling gate
#   bench           GPU: bench_decode / bench_accept / bench_prefill for both models
#   needle          GPU: 8K needle retrieval, greedy and speculative, both models
#   baseline <tag>  GPU: parity + bench + needle, bench numbers saved to results/baseline_<tag>.txt
#   compare <tag>   GPU: bench again and diff against results/baseline_<tag>.txt (flags > 1% regressions)
# Logs go to results/gate_<mode>_<timestamp>/ (gitignored).
set -u
[[ -f ${0:A:h:h}/local.env ]] && source ${0:A:h:h}/local.env
ROOT=${0:A:h:h:h}
cd $ROOT/build || exit 2
mode=${1:-static}
tag=${2:-}
stamp=$(date +%Y%m%d_%H%M%S)
out=$ROOT/results/gate_${mode}_$stamp
mkdir -p $out $ROOT/results
fails=0
run() { # name, command...
	local name=$1; shift
	if "$@" > $out/$name.log 2>&1; then echo "[ok]   $name"; else echo "[FAIL] $name (see $out/$name.log)"; fails=$((fails+1)); fi
}
gpu() { taskset -c 0-9 "$@"; }

static_checks() {
	local bad=0
	# Headers that the host-only translation units include must not name CUDA types once the backend layer exists
	for pat in cudaStream_t cuda_runtime.h; do
		local n=$(grep -rn "$pat" $ROOT/inc/engine/core $ROOT/inc/engine/runtime $ROOT/inc/engine/models 2>/dev/null | wc -l)
		echo "headers naming $pat (core, runtime, models): $n"
	done
	echo "direct <cuda_*> includes in host-only files: $(grep -rln '#include <cuda_\|#include <cublas' $ROOT/inc/engine/core $ROOT/inc/engine/runtime $ROOT/inc/engine/models $ROOT/inc/engine/loader $ROOT/inc/engine/server $ROOT/src/models/*.cpp $ROOT/src/runtime $ROOT/src/loader $ROOT/src/server $ROOT/src/core 2>/dev/null | wc -l)"
	echo "module/spec headers with <cuda_*> includes: $(grep -rln '#include <cuda_' $ROOT/inc/engine/modules $ROOT/inc/engine/spec 2>/dev/null | wc -l)"
	echo "kernel launches in models: $(grep -c '<<<' $ROOT/src/models/*.cpp 2>/dev/null | tr '\n' ' ')"
	ninja > $out/build.log 2>&1 || { echo "build FAILED"; bad=1; }
	# CPU configuration: configures without CUDA, builds, smoke test passes
	cmake -S $ROOT -B $ROOT/build_cpu -G Ninja -DENGINE_BACKEND=CPU > $out/cpu_cmake.log 2>&1 && ninja -C $ROOT/build_cpu test_cpu_smoke > $out/cpu_build.log 2>&1 && $ROOT/build_cpu/test_cpu_smoke > $out/cpu_smoke.log 2>&1 || { echo "CPU configuration FAILED (see $out/cpu_*.log)"; bad=1; }
	# CPU and Vulkan configurations: build the op-level tests and smoke tests used by `parity`
	ninja -C $ROOT/build_cpu test_cpu_smoke test_ops_linear test_ops_moe test_ops_attention > $out/cpu_ops_build.log 2>&1 || { echo "CPU op tests build FAILED"; bad=1; }
	cmake -S $ROOT -B $ROOT/build_vk -G Ninja -DENGINE_BACKEND=VULKAN > $out/vk_cmake.log 2>&1 && ninja -C $ROOT/build_vk test_vulkan_smoke test_vulkan_ref test_ops_linear test_ops_attention test_vulkan_forward > $out/vk_build.log 2>&1 || { echo "Vulkan configuration FAILED (see $out/vk_*.log)"; bad=1; }
	# Slang-generated SPIR-V must validate
	for f in slang/*.spv(N); do spirv-val $f > $out/$f:t.log 2>&1 || { echo "spirv-val FAILED: $f"; bad=1; }; done
	return $bad
}

tier2() {
	for t in test_trellis test_gemv_int8 test_mhc test_attention test_attention_prefill test_mla test_dsa test_gdn test_rows \
		test_norm test_rope test_kv_quant test_all_reduce test_moe test_nvfp4_linear test_prefill_linear test_sampling test_mla_latent test_spec_state test_slang_cpu test_slang_kernels; do
		run $t gpu ./$t
	done
	# test_dflash prints a known FATAL in its geometry step (present before 2026-10-07); judged by its final line
	run test_dflash sh -c "taskset -c 0-9 ./test_dflash | tail -3 | grep -q 'PASSED'"
	run jinja_chat_templates ./test_jinja
	run output_parser ./test_output_parser
	run tokenizer ./test_tokenizer
	run tokenizer_cases_qwen ./test_tokenizer_file $VOLTA_MODEL_ROOT/Qwen3.8-27B-exl3 --cases $ROOT/tests/data/tokenizer_cases_qwen3_5.json
	run tokenizer_cases_glm ./test_tokenizer_file $VOLTA_MODEL_ROOT/GLM-5.3-Flash-exl3-3.05bpw --cases $ROOT/tests/data/tokenizer_cases_glm5_next.json
	run backend_tiers ./test_backend_tiers $ROOT
	run ref_linear_host ./test_ref_linear
	run ref_linear_gpu gpu ./test_ref_linear --gpu
	run ref_attention_host ./test_ref_attention
	run ref_attention_gpu gpu ./test_ref_attention --gpu
	run vulkan_kernels gpu ./test_vulkan_kernels
	run vulkan_smoke gpu $ROOT/build_vk/test_vulkan_smoke
	run vulkan_ref gpu $ROOT/build_vk/test_vulkan_ref
	# Cross-backend op-level goldens: dump with the CUDA kernels, check in the CUDA build and in the CPU build
	for t in test_ops_linear test_ops_moe test_ops_attention; do
		run ${t}_dump gpu ./$t --dump $out/golden
		run ${t}_check_cuda gpu ./$t --check $out/golden
		run ${t}_check_cpu $ROOT/build_cpu/$t --check $out/golden
	done
	# Vulkan: Linear (trellis, plain), attention and the single-row GDN forms through the module API
	run test_ops_linear_check_vulkan gpu $ROOT/build_vk/test_ops_linear --check $out/golden
	run test_ops_attention_check_vulkan gpu $ROOT/build_vk/test_ops_attention --check $out/golden --only attention,gdn_single
	run coherence gpu ./test_coherence
	run prefix_restore gpu ./test_prefix_restore
	run prefix_restore_spec gpu ./test_prefix_restore --spec
	run prefix_restore_fp8 sh -c "EXL3_KV=fp8 taskset -c 0-9 ./test_prefix_restore"
	run prefix_restore_glm gpu ./test_prefix_restore --glm
	run sampling_qwen gpu ./test_sampling_gpu
	run sampling_glm gpu ./test_sampling_gpu --glm
}

benches() {
	local f=$out/bench_numbers.txt; : > $f
	for m in "" "--glm"; do
		run bench_decode$m gpu ./bench_decode $m
		grep -E "M=1 |Mode A|Mode B" $out/bench_decode$m.log | sed "s/^/decode$m: /" >> $f
	done
	run bench_accept gpu ./bench_accept
	grep -E "regime|tok/s" $out/bench_accept.log | tail -3 | sed 's/^/accept: /' >> $f
	run bench_prefill_qwen gpu ./bench_prefill 2048 512
	grep chunk $out/bench_prefill_qwen.log | sed 's/^/prefill qwen: /' >> $f
	BENCH_GLM=1 run bench_prefill_glm gpu ./bench_prefill 2048 512
	grep chunk $out/bench_prefill_glm.log | sed 's/^/prefill glm: /' >> $f
	echo "SASS HMMA count (>= 30000): $(cuobjdump -sass ./volta_server 2>/dev/null | grep -c HMMA)" >> $f
}

needles() {
	run needle_qwen gpu ./test_longctx --tokens 8000 --depth 50
	run needle_qwen_spec gpu ./test_longctx --spec --tokens 8000 --depth 70
	run needle_glm gpu ./test_longctx --glm --tokens 8000 --depth 50
	run needle_glm_spec gpu ./test_longctx --glm --spec --tokens 8000 --depth 30
}

fingerprint() { # one line per binary: md5 of per-kernel registers/stack/smem
	for b in bench_decode volta_server; do echo "$b $(cuobjdump -res-usage $b 2>/dev/null | md5sum | cut -c1-32)"; done
}

sass_fingerprint() {
	for b in bench_decode volta_server; do
		cuobjdump -sass $b 2>/dev/null | tee >(grep -c HMMA > $out/$b.hmma) | grep -E '^\s+/\*[0-9a-f]{4}\*/' | sed -E 's#/\*[0-9a-f]+\*/##;s#;.*##' | md5sum | cut -c1-32 > $out/$b.md5
		echo "$b $(cat $out/$b.md5) hmma=$(cat $out/$b.hmma)"
	done
}

compare_fp() { # name, current-file, saved-file
	if [[ ! -f $3 ]]; then cp $2 $3; echo "saved $3"; return 0; fi
	diff $3 $2 > $out/$1.diff && echo "[ok]   $1 identical" || { echo "[FAIL] $1 differs (see $out/$1.diff)"; return 1; }
}

case $mode in
	quick)
		ninja > $out/build.log 2>&1 || { echo "[FAIL] build"; fails=$((fails+1)); }
		fingerprint > $out/fingerprint.txt
		compare_fp fingerprint $out/fingerprint.txt $ROOT/results/fingerprint.txt || fails=$((fails+1))
		run coherence gpu ./test_coherence ;;
	fingerprint) fingerprint > $ROOT/results/fingerprint.txt; cat $ROOT/results/fingerprint.txt ;;
	sass)
		ninja > $out/build.log 2>&1 || { echo "[FAIL] build"; fails=$((fails+1)); }
		sass_fingerprint > $out/sass.txt; cat $out/sass.txt
		compare_fp sass $out/sass.txt $ROOT/results/sass.txt || fails=$((fails+1)) ;;
	static) static_checks || fails=$((fails+1)) ;;
	parity) tier2 ;;
	bench) benches; cat $out/bench_numbers.txt ;;
	needle) needles ;;
	baseline)
		[[ -n $tag ]] || { echo "usage: gate.sh baseline <tag>"; exit 2; }
		tier2; needles; benches
		cp $out/bench_numbers.txt $ROOT/results/baseline_$tag.txt; echo "saved results/baseline_$tag.txt" ;;
	compare)
		[[ -f $ROOT/results/baseline_$tag.txt ]] || { echo "no baseline $tag"; exit 2; }
		benches
		python3 -I - $ROOT/results/baseline_$tag.txt $out/bench_numbers.txt <<'PY'
import re, sys
def load(p):
	d = {}
	for line in open(p):
		m = re.match(r"(.*?):\s*(.*?)\s*:\s*([0-9.]+) ms", line)
		if m: d[(m.group(1), m.group(2))] = float(m.group(3))
	return d
a, b = load(sys.argv[1]), load(sys.argv[2])
bad = 0
for k in sorted(a):
	if k in b:
		r = b[k] / a[k] - 1
		flag = "REGRESSION" if r > 0.01 else "ok"
		bad += flag == "REGRESSION"
		print(f"{k[0]} {k[1]}: {a[k]:.2f} -> {b[k]:.2f} ms ({r*100:+.1f}%) {flag}")
sys.exit(1 if bad else 0)
PY
		[[ $? -eq 0 ]] || fails=$((fails+1)) ;;
	*) echo "unknown mode $mode"; exit 2 ;;
esac
echo "gate $mode: $fails failure(s); logs in $out"
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
exit $((fails > 0))
