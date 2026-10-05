cmake_minimum_required(VERSION 3.18)
project(volta_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70)

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
	src/modules/moe.cu
	src/modules/mhc.cu
	src/models/model.cpp
	src/models/qwen38_27b.cu
	src/models/glm53_flash.cu
	src/spec/dflash.cu
	src/runtime/kv_cache.cpp
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

set_target_properties(engine_kernels PROPERTIES
	ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
	OUTPUT_NAME "engine_kernels"
)

# Test Trellis Parity
add_executable(test_trellis tests/2_parity/test_trellis.cpp)
target_link_libraries(test_trellis PRIVATE engine_kernels)
set_target_properties(test_trellis PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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

add_executable(bench_nvfp4 tests/3_bench/bench_nvfp4.cpp)
target_link_libraries(bench_nvfp4 PRIVATE engine_kernels)
set_target_properties(bench_nvfp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_decode tests/3_bench/bench_decode.cpp)
target_link_libraries(bench_decode PRIVATE engine_kernels)
set_target_properties(bench_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_spec tests/3_bench/bench_spec.cpp)
target_link_libraries(bench_spec PRIVATE engine_kernels)
set_target_properties(bench_spec PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Coherence Tests
add_executable(test_coherence tests/4_coherence/test_coherence.cpp)
target_link_libraries(test_coherence PRIVATE engine_kernels)
set_target_properties(test_coherence PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Unified Production Server
add_executable(volta_server src/server/main.cpp)
target_link_libraries(volta_server PRIVATE engine_kernels pthread)
set_target_properties(volta_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
