#!/bin/zsh
# Whole-model step cost over every verify size (1 to 16 rows) at three context lengths, per model and weight format: bench_decode with
# BENCH_VL and BENCH_CTX, written as CSV (model,weights,context,rows,ms) for tests/tools/kernel_cliffs.py. The arms are lines of
# "model-flag ENGINE_TEST_<ROLE>_DIR=<dir>[;VAR=value...] label"; each waits until every GPU is at or below 55 C.
#   kernel_sweep.sh OUT.csv [ARMS_FILE]
cd ${0:A:h}/../../build
OUT=$1; ARMS=${2:-}
source ../tests/tools/models_env.sh ../tests
cool() { while nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | awk '$1>55{f=1} END{exit !f}'; do sleep 20; done }
echo "model,weights,context,rows,ms" > $OUT
while read -r flag envset label; do
  [[ -z $flag || $flag == \#* ]] && continue
  model=${label%%:*}; weights=${label#*:}
  for ctx in 8 8192 32768; do
    cool
    log=$(env ENGINE_TP=4 ENGINE_MAX_SEQ=40960 BENCH_CTX=$ctx BENCH_VL=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 ${=${envset//;/ }} numactl -N 0 -m 0 ./bench_decode ${flag/none/} 2>&1)
    echo "$log" | sed -n 's/.*M=1 decode *: *\([0-9.]*\) ms.*/\1/p' | head -1 | sed "s/^/$model,$weights,$ctx,1,/" >> $OUT
    echo "$log" | sed -n 's/.*verify block L=\([0-9]*\) : \([0-9.]*\) ms.*/\1,\2/p' | sed "s/^/$model,$weights,$ctx,/" >> $OUT
  done
done < ${ARMS:-/dev/stdin}
