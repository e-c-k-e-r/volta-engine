cmake_minimum_required(VERSION 3.18)
project(volta_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-exceptions -DNO_EXCEPTIONS")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fno-exceptions -DNO_EXCEPTIONS -DEXL3_SM70_HMMA=1 -DEXL3_VOLTA=1 --expt-relaxed-constexpr")

include_directories(
	${PROJECT_SOURCE_DIR}/inc
	${PROJECT_SOURCE_DIR}/src/kernels/trellis
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(ENGINE_CORE_SOURCES
	src/core/arena.cpp
	src/loader/safetensors.cpp
	src/comm/all_reduce_ipc.cu
	src/kernels/nvfp4/skinny_kernels.cu
	src/models/qwen38_27b.cu
	src/models/glm53_flash.cu
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

# Coherence Tests
add_executable(test_coherence tests/4_coherence/test_coherence.cpp)
target_link_libraries(test_coherence PRIVATE engine_kernels)
set_target_properties(test_coherence PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
