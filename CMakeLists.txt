cmake_minimum_required(VERSION 3.18)
project(volta_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-exceptions -DNO_EXCEPTIONS -DJSON_NOEXCEPTION")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fno-exceptions -DNO_EXCEPTIONS -DJSON_NOEXCEPTION -DEXL3_SM70_HMMA=1 -DEXL3_VOLTA=1 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -diag-suppress 20012")

include_directories(
	${PROJECT_SOURCE_DIR}/inc
	${PROJECT_SOURCE_DIR}/src/kernels/trellis
	${PROJECT_SOURCE_DIR}/deps/json/single_include
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(ENGINE_CORE_SOURCES
	src/core/arena.cpp
	src/loader/safetensors.cpp
	src/loader/nvfp4.cpp
	src/loader/tokenizer.cpp
	src/loader/chat_template.cpp
	src/comm/all_reduce_ipc.cu
	src/kernels/nvfp4/skinny_kernels.cu
	src/modules/norm.cu
	src/modules/swiglu.cu
	src/modules/rope.cu
	src/modules/attention.cu
	src/modules/gdn.cu
	src/modules/mla.cu
	src/modules/mla_latent.cu
	src/modules/dsa.cu
	src/modules/moe.cu
	src/modules/mhc.cu
	src/modules/embedding.cu
	src/models/model.cpp
	src/models/qwen38_27b.cu
	src/models/glm53_flash.cu
	src/spec/dflash_kernels.cu
	src/spec/dflash_speculator.cu
	src/spec/dflash_backbone.cu
	src/spec/tap_ring.cu
	src/runtime/kv_cache.cpp
	src/runtime/cluster.cpp
	src/runtime/dflash2_runner.cpp
	src/runtime/generator.cpp
	src/server/config.cpp
	src/server/server.cpp
)

file(GLOB TRELLIS_SOURCES
	src/kernels/trellis/*.cu
	src/kernels/trellis/comp_units/*.cu
)

add_library(engine_kernels STATIC
	${ENGINE_CORE_SOURCES}
	${TRELLIS_SOURCES}
)
# cuBLAS for the large-M prefill path (dequantize once, then a tensor-core HGEMM).
target_link_libraries(engine_kernels PUBLIC CUDA::cublas)

set_target_properties(engine_kernels PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
	OUTPUT_NAME "engine_kernels"
)

# Test Trellis Parity
add_executable(test_trellis tests/2_parity/test_trellis.cpp)
target_link_libraries(test_trellis PRIVATE engine_kernels)
set_target_properties(test_trellis PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test dp4a int8 GEMV Parity
add_executable(test_gemv_int8 tests/2_parity/test_gemv_int8.cpp)
target_link_libraries(test_gemv_int8 PRIVATE engine_kernels)
target_include_directories(test_gemv_int8 PRIVATE src/kernels/trellis)
set_target_properties(test_gemv_int8 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test mHC Parity
add_executable(test_mhc tests/2_parity/test_mhc.cpp)
target_link_libraries(test_mhc PRIVATE engine_kernels)
set_target_properties(test_mhc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test DFlash2 Parity & Spec Suite
add_executable(test_dflash tests/2_parity/test_dflash.cpp)
target_link_libraries(test_dflash PRIVATE engine_kernels)
set_target_properties(test_dflash PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test Tokenizer & Chat Template Parity
add_executable(test_tokenizer tests/2_parity/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE engine_kernels)
set_target_properties(test_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test NVLink AllReduce & Norm Parity
add_executable(test_all_reduce tests/2_parity/test_all_reduce.cpp)
target_link_libraries(test_all_reduce PRIVATE engine_kernels pthread)
set_target_properties(test_all_reduce PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test Attention Module Parity
add_executable(test_attention tests/2_parity/test_attention.cpp)
target_link_libraries(test_attention PRIVATE engine_kernels)
set_target_properties(test_attention PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test MLA Module Parity
add_executable(test_mla tests/2_parity/test_mla.cpp)
target_link_libraries(test_mla PRIVATE engine_kernels)
set_target_properties(test_mla PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test GDN Linear Attention Parity
add_executable(test_gdn tests/2_parity/test_gdn.cpp)
target_link_libraries(test_gdn PRIVATE engine_kernels)
set_target_properties(test_gdn PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test multi-row (Mode A / Mode B) forward parity against sequential M=1
add_executable(test_rows tests/2_parity/test_rows.cpp)
target_link_libraries(test_rows PRIVATE engine_kernels pthread)
set_target_properties(test_rows PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test routed MoE (persistent kernel) parity against per-expert reference
add_executable(test_moe tests/2_parity/test_moe.cpp)
target_link_libraries(test_moe PRIVATE engine_kernels)
set_target_properties(test_moe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test Norm & Elementwise Parity
add_executable(test_norm tests/2_parity/test_norm.cpp)
target_link_libraries(test_norm PRIVATE engine_kernels)
set_target_properties(test_norm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Microbenchmarks
add_executable(bench_all_reduce tests/3_bench/bench_all_reduce.cpp)
target_link_libraries(bench_all_reduce PRIVATE engine_kernels)
set_target_properties(bench_all_reduce PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_trellis tests/3_bench/bench_trellis.cpp)
target_link_libraries(bench_trellis PRIVATE engine_kernels)
set_target_properties(bench_trellis PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_shapes tests/3_bench/bench_shapes.cpp)
target_link_libraries(bench_shapes PRIVATE engine_kernels)
target_include_directories(bench_shapes PRIVATE src/kernels/trellis)
set_target_properties(bench_shapes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_nvfp4 tests/3_bench/bench_nvfp4.cpp)
target_link_libraries(bench_nvfp4 PRIVATE engine_kernels)
set_target_properties(bench_nvfp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_decode tests/3_bench/bench_decode.cpp)
target_link_libraries(bench_decode PRIVATE engine_kernels)
set_target_properties(bench_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_spec tests/3_bench/bench_spec.cpp)
target_link_libraries(bench_spec PRIVATE engine_kernels)
set_target_properties(bench_spec PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_spec_state tests/2_parity/test_spec_state.cpp)
target_link_libraries(test_spec_state PRIVATE engine_kernels)
set_target_properties(test_spec_state PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_sampling tests/2_parity/test_sampling.cpp)
target_link_libraries(test_sampling PRIVATE engine_kernels)
set_target_properties(test_sampling PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_prefill_linear tests/2_parity/test_prefill_linear.cpp)
target_link_libraries(test_prefill_linear PRIVATE engine_kernels)
set_target_properties(test_prefill_linear PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_prefill_equiv tests/2_parity/test_prefill_equiv.cpp)
target_link_libraries(test_prefill_equiv PRIVATE engine_kernels)
set_target_properties(test_prefill_equiv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_glm_rows tests/2_parity/test_glm_rows.cpp)
target_link_libraries(test_glm_rows PRIVATE engine_kernels)
set_target_properties(test_glm_rows PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_attention_prefill tests/2_parity/test_attention_prefill.cpp)
target_link_libraries(test_attention_prefill PRIVATE engine_kernels)
set_target_properties(test_attention_prefill PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_prefill tests/3_bench/bench_prefill.cpp)
target_link_libraries(bench_prefill PRIVATE engine_kernels)
set_target_properties(bench_prefill PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_accept tests/3_bench/bench_accept.cpp)
target_link_libraries(bench_accept PRIVATE engine_kernels)
set_target_properties(bench_accept PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Coherence Tests
add_executable(test_coherence tests/4_coherence/test_coherence.cpp)
target_link_libraries(test_coherence PRIVATE engine_kernels)
set_target_properties(test_coherence PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_long_context tests/4_coherence/test_long_context.cpp)
target_link_libraries(test_long_context PRIVATE engine_kernels)
set_target_properties(test_long_context PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_sampling_gpu tests/4_coherence/test_sampling_gpu.cpp)
target_link_libraries(test_sampling_gpu PRIVATE engine_kernels)
set_target_properties(test_sampling_gpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Unified Production Server
add_executable(volta_server src/server/main.cpp)
target_link_libraries(volta_server PRIVATE engine_kernels pthread)
set_target_properties(volta_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Speculative-decoding losslessness precondition: M>1 verify forward vs M=1 greedy forward
add_executable(test_spec_target_equiv tests/2_parity/test_spec_target_equiv.cpp)
target_link_libraries(test_spec_target_equiv PRIVATE engine_kernels)
set_target_properties(test_spec_target_equiv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# DFlash2 backbone building blocks parity
add_executable(test_dflash2_backbone tests/2_parity/test_dflash2_backbone.cpp)
target_link_libraries(test_dflash2_backbone PRIVATE engine_kernels)

add_executable(test_dflash2_blocks tests/2_parity/test_dflash2_blocks.cpp)
target_link_libraries(test_dflash2_blocks PRIVATE engine_kernels)
target_include_directories(test_dflash2_blocks PRIVATE src/kernels/trellis)
set_target_properties(test_dflash2_blocks PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# DFlash2 Trellis weights against the unquantized BF16 checkpoint
add_executable(test_dflash2_quant tests/2_parity/test_dflash2_quant.cpp)
target_link_libraries(test_dflash2_quant PRIVATE engine_kernels)

add_executable(test_longctx tests/4_coherence/test_longctx.cpp)
target_link_libraries(test_longctx PRIVATE engine_kernels)
set_target_properties(test_longctx PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_nvfp4_linear tests/2_parity/test_nvfp4_linear.cpp)
target_link_libraries(test_nvfp4_linear PRIVATE engine_kernels)
set_target_properties(test_nvfp4_linear PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_kv_quant tests/2_parity/test_kv_quant.cpp)
target_link_libraries(test_kv_quant PRIVATE engine_kernels)
set_target_properties(test_kv_quant PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_moe_prefill tests/3_bench/bench_moe_prefill.cpp)
target_link_libraries(bench_moe_prefill PRIVATE engine_kernels)
set_target_properties(bench_moe_prefill PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_mla_latent tests/2_parity/test_mla_latent.cpp)
target_link_libraries(test_mla_latent PRIVATE engine_kernels)
set_target_properties(test_mla_latent PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_dsa tests/2_parity/test_dsa.cpp)
target_link_libraries(test_dsa PRIVATE engine_kernels)
set_target_properties(test_dsa PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
