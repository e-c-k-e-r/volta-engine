#!/bin/bash
# Compiles each public header alone with the host C++ compiler to prove it is self-sufficient.
# Usage: tests/tools/header_check.sh   (CUDA_INCLUDE defaults to /opt/cuda/include)
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CUDA_INCLUDE="${CUDA_INCLUDE:-/opt/cuda/include}"
cd "$ROOT" || exit 1
pass=0
fail=0
for dir in modules spec core runtime models loader server; do
	for h in inc/engine/$dir/*.h; do
		[ -e "$h" ] || continue
		rel="engine/$dir/$(basename "$h")"
		if out=$(echo "#include \"$rel\"" | c++ -std=c++17 -fsyntax-only -x c++ -I inc -I src/kernels/trellis -I deps/json/single_include -I "$CUDA_INCLUDE" - 2>&1); then
			echo "PASS $rel"
			pass=$((pass + 1))
		else
			echo "FAIL $rel"
			echo "$out" | grep -m3 'error' | sed 's/^/	/'
			fail=$((fail + 1))
		fi
	done
done
echo "header_check: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
