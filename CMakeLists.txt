cmake_minimum_required(VERSION 3.18)

# Computing backend: exactly one (CUDA, CPU or VULKAN). Chosen before project() so the CPU and Vulkan configurations never enable the CUDA language.
set(ENGINE_BACKEND "CUDA" CACHE STRING "Computing backend: CUDA, CPU or VULKAN")
if(ENGINE_BACKEND STREQUAL "CUDA")
	project(volta_engine LANGUAGES CXX CUDA)
elseif(ENGINE_BACKEND STREQUAL "CPU" OR ENGINE_BACKEND STREQUAL "VULKAN")
	project(volta_engine LANGUAGES CXX)
else()
	message(FATAL_ERROR "ENGINE_BACKEND=${ENGINE_BACKEND}: only CUDA, CPU and VULKAN are implemented")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENGINE_BACKEND STREQUAL "CUDA")
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	set(CMAKE_CUDA_ARCHITECTURES 70)
	add_compile_definitions(ENGINE_BACKEND_CUDA=1)
	find_package(CUDAToolkit REQUIRED)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fno-exceptions -DNO_EXCEPTIONS -DJSON_NOEXCEPTION -DEXL3_SM70_HMMA=1 -DEXL3_VOLTA=1 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -diag-suppress 20012")
elseif(ENGINE_BACKEND STREQUAL "VULKAN")
	add_compile_definitions(ENGINE_BACKEND_VULKAN=1)
else()
	add_compile_definitions(ENGINE_BACKEND_CPU=1)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-exceptions -DNO_EXCEPTIONS -DJSON_NOEXCEPTION")

include_directories(
	${PROJECT_SOURCE_DIR}/inc
	${PROJECT_SOURCE_DIR}/src/kernels/cuda/trellis
	${PROJECT_SOURCE_DIR}/deps/json/single_include
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Host C++ (built by the host compiler; talks to the device only through engine/backend and the module headers)
set(ENGINE_HOST_SOURCES
	src/core/arena.cpp
	src/core/log.cpp
	src/loader/safetensors.cpp
	src/loader/nvfp4.cpp
	src/loader/tokenizer.cpp
	src/loader/chat_template.cpp
	src/loader/jinja.cpp
	src/loader/image.cpp
	src/loader/ngram_table.cpp
	src/models/model.cpp
	src/models/qwen3_5.cpp
	src/models/qwen3_vision.cpp
	src/models/glm5_vision.cpp
	src/models/glm5_next.cpp
	src/models/qwen4_exp.cpp
	src/runtime/options.cpp
	src/runtime/kv_cache.cpp
	src/runtime/cluster.cpp
	src/runtime/dflash2_runner.cpp
	src/runtime/mtp_runner.cpp
	src/runtime/prefix_cache.cpp
	src/runtime/generator.cpp
	src/server/config.cpp
	src/server/output_parser.cpp
	src/server/http.cpp
	src/server/chat.cpp
	src/server/openai.cpp
	src/server/anthropic.cpp
	src/server/server.cpp
)

if(ENGINE_BACKEND STREQUAL "CUDA")
# CUDA device code and its launch drivers (compiled only for the CUDA backend)
set(ENGINE_DEVICE_SOURCES
	src/comm/all_reduce_ipc.cu
	src/kernels/cuda/nvfp4/skinny_kernels.cu
	src/backend/cuda/modules/norm.cu
	src/backend/cuda/modules/swiglu.cu
	src/backend/cuda/modules/rope.cu
	src/backend/cuda/modules/attention.cu
	src/backend/cuda/modules/gdn.cu
	src/backend/cuda/modules/mla.cu
	src/backend/cuda/modules/mla_latent.cu
	src/backend/cuda/modules/dsa.cu
	src/backend/cuda/modules/moe.cu
	src/backend/cuda/modules/expert_tier.cu
	src/backend/cuda/modules/mhc.cu
	src/backend/cuda/modules/gated_residual.cu
	src/backend/cuda/modules/ple.cu
	src/backend/cuda/modules/qsa.cu
	src/backend/cuda/modules/embedding.cu
	src/backend/cuda/modules/vision.cu
	src/backend/cuda/modules/linear_trellis.cu
	src/spec/dflash_kernels.cu
	src/spec/dflash_speculator.cu
	src/spec/dflash_backbone.cu
	src/spec/tap_ring.cu
)
set(ENGINE_CORE_SOURCES ${ENGINE_HOST_SOURCES} ${ENGINE_DEVICE_SOURCES})

set(TRELLIS_SOURCES
	src/kernels/cuda/trellis/exl3_gemm.cu
	src/kernels/cuda/trellis/exl3_gemv.cu
	src/kernels/cuda/trellis/exl3_gemv_int8.cu
	src/kernels/cuda/trellis/exl3_instances.cu
	src/kernels/cuda/trellis/exl3_kernel_map.cu
	src/kernels/cuda/trellis/exl3_moe_coop.cu
	src/kernels/cuda/trellis/hadamard.cu
	src/kernels/cuda/trellis/reconstruct.cu
	src/kernels/cuda/trellis/trellis_context.cu
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

endif()

# Slang Tier B kernels (src/kernels/slang/*.slang): portable sources for the CUDA / Vulkan / CPU backends. The CUDA build keeps its
# hand-tuned kernels as the default; these are generated for the parity test and as the base for the other backends.
find_program(SLANGC slangc HINTS /opt/shader-slang/bin)
if(SLANGC)
	get_filename_component(SLANG_BIN_DIR ${SLANGC} DIRECTORY)
	set(SLANG_INCLUDE_DIR ${SLANG_BIN_DIR}/../include)
	set(SLANG_OUT ${PROJECT_BINARY_DIR}/slang)
	file(MAKE_DIRECTORY ${SLANG_OUT})
	set(SLANG_KERNELS rms_norm:rms_norm rms_norm:rms_norm_heads elementwise:add_residual elementwise:swiglu
		elementwise:swiglu_fused elementwise:embed_tokens rope:rope elementwise:moe_sum_planes elementwise:moe_accumulate
		select:batched_argmax select:router_select gdn:l2_norm_heads gdn:gated_rmsnorm
		gdn_rows:project_gates gdn_rows:causal_conv1d_silu_rows gdn_rows:causal_conv1d_silu_tiled gdn_rows:causal_conv1d_history
		gdn_rows:recurrent_step_rows kv_store:kv_store_f16 mhc:init_streams mhc:hc_apply mhc:hc_mix_partials mhc:hc_mix_finalize
		mhc:hyperhead_mean dsa:dsa_prep dsa:dsa_pool_update tap_ring:dflash_store_taps tap_ring:reduce_streams)
	set(SLANG_GENERATED "")
	foreach(item ${SLANG_KERNELS})
		string(REPLACE ":" ";" parts ${item})
		list(GET parts 0 sfile)
		list(GET parts 1 entry)
		set(ssrc ${PROJECT_SOURCE_DIR}/src/kernels/slang/${sfile}.slang)
		foreach(tgt cuda:cuda.cu:0 spirv:spv:0 cpp:cpp:1)
			string(REPLACE ":" ";" tparts ${tgt})
			list(GET tparts 0 target)
			list(GET tparts 1 ext)
			list(GET tparts 2 cpu)
			set(sout ${SLANG_OUT}/${entry}.${ext})
			add_custom_command(OUTPUT ${sout}
				COMMAND ${SLANGC} ${ssrc} -target ${target} -stage compute -entry ${entry} -DENGINE_TARGET_CPU=${cpu} -o ${sout}
				DEPENDS ${ssrc}
				COMMENT "slangc ${entry} -> ${target}")
			list(APPEND SLANG_GENERATED ${sout})
		endforeach()
	endforeach()
	add_custom_target(slang_kernels DEPENDS ${SLANG_GENERATED})

	if(ENGINE_BACKEND STREQUAL "CUDA")
		# Generated CUDA kernels live in their own translation units: the Slang prelude redefines the half2 operators
		file(GLOB SLANG_CUDA_LAUNCHERS src/kernels/slang/cuda/*.cu)
		add_library(engine_slang_cuda STATIC ${SLANG_CUDA_LAUNCHERS})
		add_dependencies(engine_slang_cuda slang_kernels)
		target_include_directories(engine_slang_cuda PRIVATE ${SLANG_OUT} ${SLANG_INCLUDE_DIR} src/kernels/slang/cuda)
		target_compile_definitions(engine_slang_cuda PRIVATE __CUDA_NO_HALF2_OPERATORS__)
		set_target_properties(engine_slang_cuda PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	endif()

	# Host target: the generated C++ kernels run on the CPU (no GPU involved)
	set(SLANG_CPU_SOURCES "")
	foreach(item ${SLANG_KERNELS})
		string(REPLACE ":" ";" parts ${item})
		list(GET parts 1 entry)
		list(APPEND SLANG_CPU_SOURCES ${SLANG_OUT}/${entry}.cpp)
	endforeach()
	set_source_files_properties(${SLANG_CPU_SOURCES} PROPERTIES GENERATED TRUE)
	add_executable(test_slang_cpu tests/2_parity/test_slang_cpu.cpp ${SLANG_CPU_SOURCES})
	add_dependencies(test_slang_cpu slang_kernels)
	target_include_directories(test_slang_cpu PRIVATE ${SLANG_INCLUDE_DIR})
	set_target_properties(test_slang_cpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	if(ENGINE_BACKEND STREQUAL "CUDA")
		add_executable(test_slang_kernels tests/2_parity/test_slang_kernels.cpp)
		target_include_directories(test_slang_kernels PRIVATE src/kernels/slang/cuda)
		target_link_libraries(test_slang_kernels PRIVATE engine_kernels engine_slang_cuda)
		set_target_properties(test_slang_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	endif()
else()
	message(STATUS "slangc not found: Slang kernels skipped")
endif()

# Capability registry check (host only)
add_executable(test_backend_tiers tests/2_parity/test_backend_tiers.cpp)
set_target_properties(test_backend_tiers PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

if(ENGINE_BACKEND STREQUAL "CUDA")
# Test Trellis Parity
add_executable(test_trellis tests/2_parity/test_trellis.cpp)
target_link_libraries(test_trellis PRIVATE engine_kernels)
set_target_properties(test_trellis PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test dp4a int8 GEMV Parity
add_executable(test_gemv_int8 tests/2_parity/test_gemv_int8.cpp)
target_link_libraries(test_gemv_int8 PRIVATE engine_kernels)
target_include_directories(test_gemv_int8 PRIVATE src/kernels/cuda/trellis)
set_target_properties(test_gemv_int8 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test RoPE Parity
add_executable(test_rope tests/2_parity/test_rope.cpp)
target_link_libraries(test_rope PRIVATE engine_kernels)
set_target_properties(test_rope PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test mHC Parity
add_executable(test_mhc tests/2_parity/test_mhc.cpp)
target_link_libraries(test_mhc PRIVATE engine_kernels)
set_target_properties(test_mhc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test DFlash2 Parity & Spec Suite
add_executable(test_dflash tests/2_parity/test_dflash.cpp)
target_link_libraries(test_dflash PRIVATE engine_kernels)
set_target_properties(test_dflash PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test Tokenizer & Chat Template Parity
add_executable(test_jinja tests/2_parity/test_jinja.cpp)
target_link_libraries(test_jinja PRIVATE engine_kernels)
target_compile_definitions(test_jinja PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
set_target_properties(test_jinja PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_image tests/2_parity/test_image.cpp)
target_link_libraries(test_image PRIVATE engine_kernels)
target_compile_definitions(test_image PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
set_target_properties(test_image PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_output_parser tests/2_parity/test_output_parser.cpp)
target_link_libraries(test_output_parser PRIVATE engine_kernels)
target_compile_definitions(test_output_parser PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
set_target_properties(test_output_parser PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_tokenizer_file tests/2_parity/test_tokenizer_file.cpp)
target_link_libraries(test_tokenizer_file PRIVATE engine_kernels)
set_target_properties(test_tokenizer_file PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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
add_executable(test_gr_tc tests/2_parity/test_gr_tc.cpp)
target_link_libraries(test_gr_tc PRIVATE engine_kernels)
set_target_properties(test_gr_tc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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
target_include_directories(bench_shapes PRIVATE src/kernels/cuda/trellis)
set_target_properties(bench_shapes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_nvfp4 tests/3_bench/bench_nvfp4.cpp)
target_link_libraries(bench_nvfp4 PRIVATE engine_kernels)
set_target_properties(bench_nvfp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_decode tests/3_bench/bench_decode.cpp)
target_link_libraries(bench_decode PRIVATE engine_kernels)
set_target_properties(bench_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_cpu_forward tests/2_parity/test_cpu_forward.cpp)
target_link_libraries(test_cpu_forward PRIVATE engine_kernels)
set_target_properties(test_cpu_forward PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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
target_include_directories(test_dflash2_blocks PRIVATE src/kernels/cuda/trellis)
set_target_properties(test_dflash2_blocks PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# DFlash2 Trellis weights against the unquantized BF16 checkpoint
add_executable(test_dflash2_quant tests/2_parity/test_dflash2_quant.cpp)
target_link_libraries(test_dflash2_quant PRIVATE engine_kernels)

add_executable(test_vision tests/4_coherence/test_vision.cpp)
target_link_libraries(test_vision PRIVATE engine_kernels)
target_compile_definitions(test_vision PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
set_target_properties(test_vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_mtp tests/4_coherence/test_mtp.cpp)
target_link_libraries(test_mtp PRIVATE engine_kernels)
set_target_properties(test_mtp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_prefix_restore tests/4_coherence/test_prefix_restore.cpp)
target_link_libraries(test_prefix_restore PRIVATE engine_kernels)
set_target_properties(test_prefix_restore PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_qwen4_exp tests/4_coherence/test_qwen4_exp.cpp)
target_link_libraries(test_qwen4_exp PRIVATE engine_kernels)
set_target_properties(test_qwen4_exp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_chunk_boundary tests/4_coherence/test_chunk_boundary.cpp)
target_link_libraries(test_chunk_boundary PRIVATE engine_kernels)
set_target_properties(test_chunk_boundary PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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
add_executable(bench_moe_routes tests/3_bench/bench_moe_routes.cpp)
target_link_libraries(bench_moe_routes PRIVATE engine_kernels)
set_target_properties(bench_moe_routes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_mla_latent tests/2_parity/test_mla_latent.cpp)
target_link_libraries(test_mla_latent PRIVATE engine_kernels)
set_target_properties(test_mla_latent PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_dsa tests/2_parity/test_dsa.cpp)
target_link_libraries(test_dsa PRIVATE engine_kernels)
set_target_properties(test_dsa PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_expert_heat tests/3_bench/bench_expert_heat.cpp)
target_link_libraries(bench_expert_heat PRIVATE engine_kernels)
set_target_properties(bench_expert_heat PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()

# Per-lane build fragments (each owned by one Wave 2 lane; empty until the lane fills it in)
include(${PROJECT_SOURCE_DIR}/cmake/cpu_backend.cmake OPTIONAL)
include(${PROJECT_SOURCE_DIR}/cmake/reference_kernels.cmake OPTIONAL)
# The CPU configuration has no Vulkan targets: they compile against the other backend definitions and break all-targets builds
if(NOT ENGINE_BACKEND STREQUAL "CPU")
	include(${PROJECT_SOURCE_DIR}/cmake/vulkan_backend.cmake OPTIONAL)
endif()
include(${PROJECT_SOURCE_DIR}/cmake/ops_tests_linear.cmake OPTIONAL)
include(${PROJECT_SOURCE_DIR}/cmake/ops_tests_attention.cmake OPTIONAL)
