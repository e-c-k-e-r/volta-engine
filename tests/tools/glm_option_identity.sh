#!/bin/zsh
# GLM decode options that must not change a bit: test_glm_rows (run from the build directory) is hashed with the defaults, with side_streams off and with hc_fuse
# off, in the NVFP4 dense configuration (the overlaps and the fused all-reduce only apply to NVFP4 / FP16 projections and decode-sized rows).
# Needs ENGINE_MODEL_ROOT with GLM-5.3-Flash-exl3-3.05bpw, GLM-5.3-Flash-NVFP4 and GLM-5.3-Flash-NVFP4-dense.
M=${ENGINE_MODEL_ROOT:?ENGINE_MODEL_ROOT is not set}
base=(ENGINE_NVFP4_DENSE_SRC=$M/GLM-5.3-Flash-NVFP4 ENGINE_NVFP4_DENSE_DIR=$M/GLM-5.3-Flash-NVFP4-dense ENGINE_NVFP4_DENSE=kda,mla,idx,head,dense,shared ENGINE_EXPERT_CACHE=0.9)
h() { env $base "$@" taskset -c 0-9 ./test_glm_rows 2>&1 | sed -n '/100-token/,$p' | md5sum | cut -d' ' -f1; }
a=$(h)
b=$(h ENGINE_DEBUG_SIDE_STREAMS=0)
c=$(h ENGINE_DEBUG_HC_FUSE=0)
echo "defaults $a"
echo "side_streams=0 $b"
echo "hc_fuse=0 $c"
[[ -n $a && $a == $b && $a == $c ]] || { echo "FAIL: the outputs differ"; exit 1; }
echo "identical"
