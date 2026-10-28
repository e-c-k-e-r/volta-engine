cmake_minimum_required(VERSION 3.18)

# Computing backend: exactly one (CUDA, ROCM, CPU or VULKAN). Chosen before project() so the CPU and Vulkan configurations never enable the CUDA language.
# ROCM compiles the CUDA sources as HIP (src/backend/rocm: compat.h, force-included, and stand-ins for the CUDA headers).
set(ENGINE_BACKEND "CUDA" CACHE STRING "Computing backend: CUDA, ROCM, CPU or VULKAN")

# CUDA targets, set before project() so CMake does not substitute the compiler's default. 70 (V100) is the tuned target; 86 (RTX 30xx /
# A10 / A40) and 89 (RTX 40xx / L4 / L40) build the same sources, taking the native mma.m16n8k16 / ldmatrix / cp.async paths where the
# kernels have them. Several at once give one fat binary: -DCMAKE_CUDA_ARCHITECTURES="70;86;89". CUDA 13 dropped 70.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures to build for (70, 86, 89, a ;-list of them, or native)")
endif()
# ROCM targets: RDNA, wave32 (every kernel is written for 32-lane warps). gfx1100 = RX 7900 XT / XTX; also gfx1101 / gfx1102 (RX 7800 /
# 7600), gfx1200 / gfx1201 (RX 9000 series). A ;-list builds one binary for several
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
	set(CMAKE_HIP_ARCHITECTURES gfx1100 CACHE STRING "ROCm targets to build for (RDNA, wave32: gfx1100, gfx1101, gfx1201, ...)")
endif()
if(ENGINE_BACKEND STREQUAL "CUDA")
	project(volta_engine LANGUAGES CXX CUDA)
elseif(ENGINE_BACKEND STREQUAL "ROCM")
	project(volta_engine LANGUAGES CXX HIP)
elseif(ENGINE_BACKEND STREQUAL "CPU" OR ENGINE_BACKEND STREQUAL "VULKAN")
	project(volta_engine LANGUAGES CXX)
else()
	message(FATAL_ERROR "ENGINE_BACKEND=${ENGINE_BACKEND}: only CUDA, ROCM, CPU and VULKAN are implemented")
endif()
# The GPU configurations share every target below; ENGINE_BACKEND_CUDA (the code's switch for that path) is defined for both
if(ENGINE_BACKEND STREQUAL "CUDA" OR ENGINE_BACKEND STREQUAL "ROCM")
	set(ENGINE_GPU ON)
else()
	set(ENGINE_GPU OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ENGINE_BACKEND STREQUAL "CUDA")
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	add_compile_definitions(ENGINE_BACKEND_CUDA=1)
	find_package(CUDAToolkit REQUIRED)
	if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13 AND "70" IN_LIST CMAKE_CUDA_ARCHITECTURES)
		message(FATAL_ERROR "CUDA ${CUDAToolkit_VERSION} cannot target SM_70: use CUDA 12.x, or drop 70 from CMAKE_CUDA_ARCHITECTURES")
	endif()
	message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
	# The EXL3 GEMM uses the Volta m8n8k4 HMMA inner loop on SM_70 and the upstream m16n8k16 one on SM_80+ (trellis/arch.cuh). ON forces the
	# Volta loop on every architecture (m8n8k4 is legal PTX on SM_80+, just not native): a fallback if the native loop misbehaves on a card
	option(ENGINE_EXL3_FORCE_VOLTA_MMA "EXL3 GEMM: use the Volta m8n8k4 inner loop on every architecture" OFF)
	set(ENGINE_EXL3_DEFS "-DEXL3_SM70_HMMA=1")
	if(ENGINE_EXL3_FORCE_VOLTA_MMA)
		set(ENGINE_EXL3_DEFS "${ENGINE_EXL3_DEFS} -DEXL3_VOLTA=1")
	endif()
	# Proof of concept: read the weight tiles in the Volta order of tests/tools/exl3_volta_layout.py (no fragment shuffles in the GEMM). Such a build
	# only runs checkpoints written in that order, and only the dense GEMM paths (gemm kernels, reconstruct) know it
	option(ENGINE_EXL3_VOLTA_LAYOUT "EXL3 GEMM: weight tiles in the Volta order (proof of concept)" OFF)
	if(ENGINE_EXL3_VOLTA_LAYOUT)
		set(ENGINE_EXL3_DEFS "${ENGINE_EXL3_DEFS} -DEXL3_VOLTA_LAYOUT=1")
		add_compile_definitions(ENGINE_BUILD_EXL3_VOLTA_LAYOUT=1)    # the loader refuses a checkpoint written in the other tile order
	endif()
	# Hardware overrides (0: query the device at startup). ENGINE_SM_COUNT caps the SMs the persistent / cooperative kernels spread over;
	# ENGINE_MAX_SMEM is the opt-in shared memory per block the big kernels request (default 98304, which V100, SM_86 and SM_89 all allow)
	set(ENGINE_SM_COUNT 0 CACHE STRING "SMs to size persistent / cooperative grids to (0: the device's own count)")
	set(ENGINE_MAX_SMEM 0 CACHE STRING "Opt-in dynamic shared memory per block in bytes (0: the default 98304)")
	if(NOT ENGINE_SM_COUNT EQUAL 0)
		add_compile_definitions(ENGINE_SM_COUNT=${ENGINE_SM_COUNT})
	endif()
	if(NOT ENGINE_MAX_SMEM EQUAL 0)
		add_compile_definitions(ENGINE_MAX_SMEM=${ENGINE_MAX_SMEM})
	endif()
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fno-exceptions -DNO_EXCEPTIONS -DJSON_NOEXCEPTION ${ENGINE_EXL3_DEFS} --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -diag-suppress 20012")
elseif(ENGINE_BACKEND STREQUAL "ROCM")
	set(CMAKE_HIP_STANDARD 17)
	set(CMAKE_HIP_STANDARD_REQUIRED ON)
	foreach(arch ${CMAKE_HIP_ARCHITECTURES})
		if(NOT arch MATCHES "^gfx1[0-9]")
			message(FATAL_ERROR "ROCM target ${arch}: the kernels assume wave32, which only RDNA (gfx10xx / gfx11xx / gfx12xx) runs; CDNA / GCN (gfx9xx) are wave64")
		endif()
	endforeach()
	message(STATUS "ROCm targets: ${CMAKE_HIP_ARCHITECTURES}")
	find_package(hip REQUIRED)
	find_package(hipblas REQUIRED)
	# Every target: the CUDA path's code with the HIP runtime under it. compat.h maps the CUDA names, src/backend/rocm/include stands
	# in for the CUDA headers; hip::host gives host translation units the HIP headers and platform definitions
	add_compile_definitions(ENGINE_BACKEND_CUDA=1 ENGINE_BACKEND_ROCM=1 USE_ROCM=1)
	include_directories(BEFORE ${PROJECT_SOURCE_DIR}/src/backend/rocm/include)
	add_compile_options(-include ${PROJECT_SOURCE_DIR}/src/backend/rocm/compat.h)
	link_libraries(hip::host)
	# RDNA grants a workgroup 64 KB of LDS (V100 96 KB): the kernels that need more are routed to smaller ones on ROCm, and the EXL3
	# GEMM picks among the shapes that fit
	set(ENGINE_SM_COUNT 0 CACHE STRING "Compute units to size persistent / cooperative grids to (0: the device's own count)")
	set(ENGINE_MAX_SMEM 65536 CACHE STRING "Opt-in dynamic shared memory per block in bytes")
	if(NOT ENGINE_SM_COUNT EQUAL 0)
		add_compile_definitions(ENGINE_SM_COUNT=${ENGINE_SM_COUNT})
	endif()
	add_compile_definitions(ENGINE_MAX_SMEM=${ENGINE_MAX_SMEM})
	set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -fno-exceptions -DNO_EXCEPTIONS -DJSON_NOEXCEPTION -DEXL3_SM70_HMMA=1 -Wno-unused-result -Wno-register")
elseif(ENGINE_BACKEND STREQUAL "VULKAN")
	add_compile_definitions(ENGINE_BACKEND_VULKAN=1)
else()
	add_compile_definitions(ENGINE_BACKEND_CPU=1)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-exceptions -DNO_EXCEPTIONS -DJSON_NOEXCEPTION")

# The CPU backend is the speed of the host's FP16 conversions and FMAs (the reference kernels convert every operand); the baseline x86-64 target has neither
option(ENGINE_CPU_NATIVE "CPU backend: compile for the build host (-march=native)" ON)
if(ENGINE_BACKEND STREQUAL "CPU" AND ENGINE_CPU_NATIVE)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

include_directories(
	${PROJECT_SOURCE_DIR}/inc
	${PROJECT_SOURCE_DIR}/src/kernels/cuda/trellis
	${PROJECT_SOURCE_DIR}/deps/json/single_include
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Terminal dashboard (volta_top and the server's --tui); host C++ only, no vendor runtime
set(ENGINE_TUI_SOURCES
	src/server/tui/term.cpp
	src/server/tui/view.cpp
	src/server/tui/app.cpp
)

# Host C++ (built by the host compiler; talks to the device only through engine/backend and the module headers)
set(ENGINE_HOST_SOURCES
	src/core/arena.cpp
	src/core/log.cpp
	src/loader/safetensors.cpp
	src/loader/nvfp4.cpp
	src/loader/tokenizer.cpp
	src/video/h3_vae_config.cpp
	src/video/h3_image.cpp
	src/video/h3_refs.cpp
	src/loader/chat_template.cpp
	src/loader/deepseek_encoding.cpp
	src/loader/jinja.cpp
	src/loader/image.cpp
	src/loader/ngram_table.cpp
	src/models/model.cpp
	src/models/qwen3_5.cpp
	src/models/steer_host.cpp
	src/models/qwen3_vision.cpp
	src/models/glm5_vision.cpp
	src/models/deepseek_v41_vision.cpp
	src/models/glm5_next.cpp
	src/models/qwen4_exp.cpp
	src/models/deepseek_v41.cpp
	src/runtime/options.cpp
	src/runtime/kv_cache.cpp
	src/runtime/cluster.cpp
	src/runtime/dflash2_runner.cpp
	src/runtime/mtp_runner.cpp
	src/runtime/suffix_runner.cpp
	src/runtime/dspark_runner.cpp
	src/runtime/prefix_cache.cpp
	src/runtime/steer_file.cpp
	src/runtime/steer_probe.cpp
	src/runtime/generator_probe.cpp
	src/runtime/cpu_experts.cpp
	src/runtime/cpu_expert_service.cpp
	src/runtime/cpu_expert_attach.cpp
	src/kernels/cpu/cpu_util.cpp
	src/kernels/cpu/pool.cpp
	src/kernels/cpu/trellis_linear.cpp
	src/kernels/cpu/nvfp4_linear.cpp
	src/runtime/generator.cpp
	src/server/config.cpp
	src/server/output_parser.cpp
	src/server/http.cpp
	src/server/chat.cpp
	src/server/openai.cpp
	src/server/anthropic.cpp
	src/server/server.cpp
	src/server/monitor.cpp
	src/server/monitor_page.cpp
	src/server/system_stats.cpp
	${ENGINE_TUI_SOURCES}
)

if(ENGINE_GPU)
# CUDA device code and its launch drivers (compiled only for the GPU backends; as HIP in the ROCM configuration)
set(ENGINE_DEVICE_SOURCES
	src/server/video/adapters.cpp
	src/server/video/jobs.cpp
	src/server/video/service.cpp
	src/server/video/api.cpp
	src/server/video/metrics.cpp
	src/server/video/page.cpp
	src/comm/all_reduce_ipc.cu
	src/kernels/cuda/nvfp4/skinny_kernels.cu
	src/kernels/cuda/nvfp4/moe_kernels.cu
	src/backend/cuda/modules/norm.cu
	src/backend/cuda/modules/swiglu.cu
	src/backend/cuda/modules/rope.cu
	src/backend/cuda/modules/attention.cu
	src/backend/cuda/modules/gdn.cu
	src/backend/cuda/modules/mla_latent.cu
	src/backend/cuda/modules/dsa.cu
	src/backend/cuda/modules/moe.cu
	src/backend/cuda/modules/expert_tier.cu
	src/backend/cuda/modules/cpu_experts.cu
	src/backend/cuda/modules/mhc.cu
	src/backend/cuda/modules/gated_residual.cu
	src/backend/cuda/modules/ple.cu
	src/backend/cuda/modules/hessian.cu
	src/backend/cuda/modules/steer.cu
	src/backend/cuda/modules/qsa.cu
	src/backend/cuda/modules/dsv4.cu
	src/backend/cuda/modules/embedding.cu
	src/backend/cuda/modules/vision.cu
	src/backend/cuda/modules/linear_trellis.cu
	src/video/h3_dit.cu
	src/video/h3_vae.cu
	src/video/h3_audio.cu
	src/video/h3_vae_enc.cu
	src/video/h3_text.cu
	src/video/h3_vision.cu
	src/video/h3_dit_group.cpp
	src/video/h3_text_group.cpp
	src/video/h3_caption.cpp
	src/video/h3_check.cpp
	src/video/h3_pipeline.cu
	src/video/h3_engine.cpp
	src/video/h3_stage_refs.cpp
	src/video/h3_stage_text.cpp
	src/video/h3_stage_denoise.cpp
	src/video/h3_stage_decode.cu
	src/kernels/cuda/h3/h3_attn.cu
	src/spec/dflash_kernels.cu
	src/spec/dflash_speculator.cu
	src/spec/dflash_backbone.cu
	src/spec/tap_ring.cu
)
set(ENGINE_CORE_SOURCES ${ENGINE_HOST_SOURCES} ${ENGINE_DEVICE_SOURCES})

# EXL3 GEMM instantiations: one generated translation unit per (bit width, codebook) so the template-heavy kernels compile in parallel
# (exl3_instances.cu alone took 450 s). Codebook 0 is 3inst, 1 mcg, 2 mul1; every checkpoint is made with one of them.
set(ENGINE_EXL3_CODEBOOKS "0;1;2" CACHE STRING "EXL3 codebooks to build GEMM kernels for (0: 3inst, 1: mcg, 2: mul1). A checkpoint using a codebook that is left out fails to start; fewer codebooks compile proportionally faster")
set(ENGINE_EXL3_PRUNE "AUTO" CACHE STRING "CUDA: leave out the EXL3 GEMM tile shapes the built architectures never select (shape 1 below SM_90, shape 4 at 7 and 8 bits below SM_89); selection steps down to a shape that exists. AUTO: on when the only target is SM_70, otherwise off. ON / OFF force it")
set_property(CACHE ENGINE_EXL3_PRUNE PROPERTY STRINGS AUTO ON OFF)
set(EXL3_UNIT_SOURCES "")
foreach(EXL3_UNIT_CB 0 1 2)
	if(EXL3_UNIT_CB IN_LIST ENGINE_EXL3_CODEBOOKS)
		set(EXL3_UNIT_BITS_LIST 2 3 4 5 6 7 8)
		if(EXL3_UNIT_CB EQUAL 2)
			list(PREPEND EXL3_UNIT_BITS_LIST 1)   # 1-bit weights exist with the mul1 codebook only (the DeepSeek-V4.1 SAGE packs' experts)
		endif()
		foreach(EXL3_UNIT_BITS ${EXL3_UNIT_BITS_LIST})
			set(unit "${PROJECT_BINARY_DIR}/exl3_units/exl3_inst_b${EXL3_UNIT_BITS}_cb${EXL3_UNIT_CB}.cu")
			configure_file(src/kernels/cuda/trellis/exl3_instance_unit.cu.in "${unit}" @ONLY)
			list(APPEND EXL3_UNIT_SOURCES "${unit}")
		endforeach()
	else()
		add_compile_definitions(EXL3_CB_BUILT_${EXL3_UNIT_CB}=0)
	endif()
endforeach()
if(ENGINE_BACKEND STREQUAL "CUDA")
	# The targets actually built: "native" resolves to the installed GPUs (CMake detects them when the CUDA language is enabled)
	set(EXL3_ARCHS ${CMAKE_CUDA_ARCHITECTURES})
	if("native" IN_LIST EXL3_ARCHS AND DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE)
		list(REMOVE_ITEM EXL3_ARCHS native)
		list(APPEND EXL3_ARCHS ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
	endif()
	set(EXL3_SHAPE1 0)
	set(EXL3_SHAPE4_HIGH_K 0)
	set(EXL3_VOLTA_ONLY TRUE)
	foreach(arch ${EXL3_ARCHS})
		string(REGEX REPLACE "-(real|virtual)$" "" arch_num "${arch}")
		if(NOT arch_num MATCHES "^[0-9]+$" OR arch_num GREATER_EQUAL 90)
			set(EXL3_SHAPE1 1)
			set(EXL3_SHAPE4_HIGH_K 1)
		elseif(arch_num GREATER_EQUAL 89)
			set(EXL3_SHAPE4_HIGH_K 1)
		endif()
		if(NOT arch_num STREQUAL "70")
			set(EXL3_VOLTA_ONLY FALSE)
		endif()
	endforeach()
	if(ENGINE_EXL3_PRUNE STREQUAL "ON" OR (ENGINE_EXL3_PRUNE STREQUAL "AUTO" AND EXL3_VOLTA_ONLY))
		add_compile_definitions(EXL3_BUILD_SHAPE1=${EXL3_SHAPE1} EXL3_BUILD_SHAPE4_HIGH_K=${EXL3_SHAPE4_HIGH_K})
		message(STATUS "EXL3 pruning (${EXL3_ARCHS}): shape 1 ${EXL3_SHAPE1}, shape 4 at 7 and 8 bits ${EXL3_SHAPE4_HIGH_K}")
	endif()
endif()

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
if(ENGINE_BACKEND STREQUAL "ROCM")
	list(APPEND TRELLIS_SOURCES src/kernels/rocm/trellis/exl3_gemv.cu src/kernels/rocm/trellis/exl3_gemm.cu src/kernels/rocm/trellis/exl3_moe.cu)
endif()
list(APPEND TRELLIS_SOURCES ${EXL3_UNIT_SOURCES})

if(ENGINE_BACKEND STREQUAL "ROCM")
	set(ENGINE_ROCM_HIP_SOURCES ${ENGINE_CORE_SOURCES} ${TRELLIS_SOURCES})
	list(FILTER ENGINE_ROCM_HIP_SOURCES INCLUDE REGEX "\\.cu$")
	set_source_files_properties(${ENGINE_ROCM_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
endif()
add_library(engine_kernels STATIC
	${ENGINE_CORE_SOURCES}
	${TRELLIS_SOURCES}
)
# cuBLAS (hipBLAS on ROCm) for the large-M prefill path (dequantize once, then a tensor-core HGEMM).
if(ENGINE_BACKEND STREQUAL "ROCM")
	target_link_libraries(engine_kernels PUBLIC roc::hipblas hip::host)
else()
	target_link_libraries(engine_kernels PUBLIC CUDA::cublas)
endif()

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
		gdn_rows:recurrent_step_rows kv_store:kv_store_f16 kv_store:kv_store_q mhc:init_streams mhc:hc_apply mhc:hc_mix_partials mhc:hc_mix_finalize
		mhc:hyperhead_mean dsa:dsa_prep dsa:dsa_pool_update tap_ring:dflash_store_taps tap_ring:reduce_streams)
	set(SLANG_COMMON ${PROJECT_SOURCE_DIR}/src/kernels/slang/kv_codec.slang)
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
				DEPENDS ${ssrc} ${SLANG_COMMON}
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

# Tensor-core fragment maps (mma.m16n8k16 / m8n8k4 on a simulated warp; host only, no GPU)
add_executable(test_mma_frag tests/2_parity/test_mma_frag.cpp)
set_target_properties(test_mma_frag PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

if(ENGINE_GPU)
# Test Trellis Parity
add_executable(test_trellis tests/2_parity/test_trellis.cpp)
target_link_libraries(test_trellis PRIVATE engine_kernels)
set_target_properties(test_trellis PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test the trellis pair decode bit for bit (every index, the three codebooks; on ROCm its RDNA form)
add_executable(test_trellis_decode tests/2_parity/test_trellis_decode.cu)
target_include_directories(test_trellis_decode PRIVATE ${PROJECT_SOURCE_DIR}/src/kernels/cuda/trellis)
if(ENGINE_BACKEND STREQUAL "ROCM")
	set_source_files_properties(tests/2_parity/test_trellis_decode.cu PROPERTIES LANGUAGE HIP)
endif()
set_target_properties(test_trellis_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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
add_executable(test_dsv41_encoding tests/2_parity/test_dsv41_encoding.cpp)
target_link_libraries(test_dsv41_encoding PRIVATE engine_kernels)
set_target_properties(test_dsv41_encoding PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_dsv41_vision tests/2_parity/test_dsv41_vision.cpp)
target_link_libraries(test_dsv41_vision PRIVATE engine_kernels)
set_target_properties(test_dsv41_vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_jinja tests/2_parity/test_jinja.cpp)
target_link_libraries(test_jinja PRIVATE engine_kernels)
target_compile_definitions(test_jinja PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
set_target_properties(test_jinja PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_image tests/2_parity/test_image.cpp)
target_link_libraries(test_image PRIVATE engine_kernels)
target_compile_definitions(test_image PRIVATE TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
set_target_properties(test_image PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_trellis_layout tests/2_parity/test_trellis_layout.cpp)
target_link_libraries(test_trellis_layout PRIVATE engine_kernels)
set_target_properties(test_trellis_layout PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(diag_logits_dump tests/4_coherence/diag_logits_dump.cpp)
target_link_libraries(diag_logits_dump PRIVATE engine_kernels)
set_target_properties(diag_logits_dump PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_tui_view tests/2_parity/test_tui_view.cpp)
target_link_libraries(test_tui_view PRIVATE engine_kernels)
set_target_properties(test_tui_view PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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

# Test GDN Linear Attention Parity
add_executable(test_gdn tests/2_parity/test_gdn.cpp)
target_link_libraries(test_gdn PRIVATE engine_kernels)
set_target_properties(test_gdn PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test multi-row (Mode A / Mode B) forward parity against sequential M=1
add_executable(test_rows tests/2_parity/test_rows.cpp)
target_link_libraries(test_rows PRIVATE engine_kernels pthread)
set_target_properties(test_rows PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(diag_logits tests/4_coherence/diag_logits.cpp)
target_link_libraries(diag_logits PRIVATE engine_kernels pthread)
set_target_properties(diag_logits PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(diag_glm_tie tests/4_coherence/diag_glm_tie.cpp)
target_link_libraries(diag_glm_tie PRIVATE engine_kernels pthread)
set_target_properties(diag_glm_tie PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_mixed_rows tests/4_coherence/test_mixed_rows.cpp)
target_link_libraries(test_mixed_rows PRIVATE engine_kernels pthread)
set_target_properties(test_mixed_rows PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# Test routed MoE (persistent kernel) parity against per-expert reference
add_executable(test_moe tests/2_parity/test_moe.cpp)
target_link_libraries(test_moe PRIVATE engine_kernels)
set_target_properties(test_moe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_glm_qkv tests/2_parity/test_glm_qkv.cpp)
target_link_libraries(test_glm_qkv PRIVATE engine_kernels)
set_target_properties(test_glm_qkv PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_gr_tc tests/2_parity/test_gr_tc.cpp)
target_link_libraries(test_gr_tc PRIVATE engine_kernels)
set_target_properties(test_gr_tc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_steer_probe tests/2_parity/test_steer_probe.cpp)
target_link_libraries(test_steer_probe PRIVATE engine_kernels)
set_target_properties(test_steer_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_steer tests/2_parity/test_steer.cpp)
target_link_libraries(test_steer PRIVATE engine_kernels)
set_target_properties(test_steer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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

add_executable(test_dsv41 tests/4_coherence/test_dsv41.cpp)
target_link_libraries(test_dsv41 PRIVATE engine_kernels)
set_target_properties(test_dsv41 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")


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
add_executable(test_steer_model tests/4_coherence/test_steer_model.cpp)
target_link_libraries(test_steer_model PRIVATE engine_kernels)
set_target_properties(test_steer_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(steer_build tests/tools/steer_build.cpp)
target_link_libraries(steer_build PRIVATE engine_kernels)
set_target_properties(steer_build PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_ppl tests/4_coherence/test_ppl.cpp)
target_link_libraries(test_ppl PRIVATE engine_kernels)
set_target_properties(test_ppl PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

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

add_executable(test_dsv4_indexer tests/2_parity/test_dsv4_indexer.cpp)
target_link_libraries(test_dsv4_indexer PRIVATE engine_kernels)
set_target_properties(test_dsv4_indexer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_dsv4_fp8 tests/2_parity/test_dsv4_fp8.cpp)
target_link_libraries(test_dsv4_fp8 PRIVATE engine_kernels)
set_target_properties(test_dsv4_fp8 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_dsa tests/2_parity/test_dsa.cpp)
target_link_libraries(test_dsa PRIVATE engine_kernels)
set_target_properties(test_dsa PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_cpu_experts tests/2_parity/test_cpu_experts.cpp)
target_link_libraries(test_cpu_experts PRIVATE engine_kernels)
set_target_properties(test_cpu_experts PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_cpu_bridge tests/2_parity/test_cpu_bridge.cpp)
target_link_libraries(test_cpu_bridge PRIVATE engine_kernels)
set_target_properties(test_cpu_bridge PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_cpu_nvfp4_vs_gpu tests/2_parity/test_cpu_nvfp4_vs_gpu.cpp)
target_link_libraries(test_cpu_nvfp4_vs_gpu PRIVATE engine_kernels)
set_target_properties(test_cpu_nvfp4_vs_gpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_cpu_trellis_vs_gpu tests/2_parity/test_cpu_trellis_vs_gpu.cpp)
target_link_libraries(test_cpu_trellis_vs_gpu PRIVATE engine_kernels)
set_target_properties(test_cpu_trellis_vs_gpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_kv_pages tests/2_parity/test_kv_pages.cpp)
target_link_libraries(test_kv_pages PRIVATE engine_kernels)
set_target_properties(test_kv_pages PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(bench_expert_heat tests/3_bench/bench_expert_heat.cpp)
target_link_libraries(bench_expert_heat PRIVATE engine_kernels)
set_target_properties(bench_expert_heat PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(bench_router tests/3_bench/bench_router.cpp)
target_link_libraries(bench_router PRIVATE engine_kernels)
set_target_properties(bench_router PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(bench_kda tests/3_bench/bench_kda.cpp)
target_link_libraries(bench_kda PRIVATE engine_kernels)
set_target_properties(bench_kda PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(bench_mhc tests/3_bench/bench_mhc.cpp)
target_link_libraries(bench_mhc PRIVATE engine_kernels)
set_target_properties(bench_mhc PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_gr_decode tests/2_parity/test_gr_decode.cpp)
target_link_libraries(test_gr_decode PRIVATE engine_kernels)
set_target_properties(test_gr_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(bench_nvfp4_shapes tests/3_bench/bench_nvfp4_shapes.cpp)
target_link_libraries(bench_nvfp4_shapes PRIVATE engine_kernels)
set_target_properties(bench_nvfp4_shapes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(bench_nvfp4_moe tests/3_bench/bench_nvfp4_moe.cpp)
target_link_libraries(bench_nvfp4_moe PRIVATE engine_kernels)
set_target_properties(bench_nvfp4_moe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_nvfp4_quantize tests/2_parity/test_nvfp4_quantize.cpp)
target_link_libraries(test_nvfp4_quantize PRIVATE engine_kernels)
set_target_properties(test_nvfp4_quantize PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_executable(test_sparse_attn tests/2_parity/test_sparse_attn.cpp)
target_link_libraries(test_sparse_attn PRIVATE engine_kernels)
set_target_properties(test_sparse_attn PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()

# Per-lane build fragments (each owned by one Wave 2 lane; empty until the lane fills it in)
# volta_top: the dashboard as a stand-alone client of a running server's /api/stats (the same on every backend)
add_executable(volta_top src/server/tui/top_main.cpp src/server/tui/http_get.cpp ${ENGINE_TUI_SOURCES})
set_target_properties(volta_top PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

include(${PROJECT_SOURCE_DIR}/cmake/cpu_backend.cmake OPTIONAL)
include(${PROJECT_SOURCE_DIR}/cmake/reference_kernels.cmake OPTIONAL)
# The CPU configuration has no Vulkan targets: they compile against the other backend definitions and break all-targets builds
if(NOT ENGINE_BACKEND STREQUAL "CPU")
	include(${PROJECT_SOURCE_DIR}/cmake/vulkan_backend.cmake OPTIONAL)
endif()
include(${PROJECT_SOURCE_DIR}/cmake/ops_tests_linear.cmake OPTIONAL)
include(${PROJECT_SOURCE_DIR}/cmake/ops_tests_attention.cmake OPTIONAL)
# MiniMax-H3: parity tests, benches and the CLI (GPU configurations; on ROCm it builds but is not supported, see docs/GETTING_STARTED.md)
if(ENGINE_GPU)
	set(H3_TOOLS
		tests/3_bench/bench_h3_attn.cpp
		tests/3_bench/bench_h3_dit.cpp
		tests/3_bench/bench_h3_gemm.cpp
		tests/3_bench/h3_text_repeat.cpp
		tests/3_bench/p2p_check.cpp
		tests/2_parity/test_h3_attn.cpp
		tests/2_parity/test_h3_audio.cpp
		tests/2_parity/test_h3_audio_enc.cpp
		tests/2_parity/test_h3_dit.cpp
		tests/2_parity/test_h3_dit_sp.cpp
		tests/2_parity/test_h3_dit_tp.cpp
		tests/2_parity/test_h3_image.cpp
		tests/2_parity/test_h3_layout.cpp
		tests/2_parity/test_h3_ref_layout.cpp
		tests/2_parity/test_h3_caption.cpp
		tests/2_parity/test_h3_text.cpp
		tests/2_parity/test_h3_text_identity.cpp
		tests/2_parity/test_h3_text_layers.cpp
		tests/2_parity/test_h3_text_mm.cpp
		tests/2_parity/test_h3_text_tp.cpp
		tests/2_parity/test_h3_vae.cpp
		tests/2_parity/test_h3_vae_det.cpp
		tests/2_parity/test_h3_vae_enc.cpp
		tests/2_parity/test_h3_vision.cpp
	)
	foreach(src IN LISTS H3_TOOLS)
		get_filename_component(tool ${src} NAME_WE)
		add_executable(${tool} ${src})
		target_link_libraries(${tool} PRIVATE engine_kernels)
		set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	endforeach()
	add_executable(h3_generate src/video/h3_generate_main.cpp)
	target_link_libraries(h3_generate PRIVATE engine_kernels)
	set_target_properties(h3_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	# NCCL is optional: it enables tensor parallelism (the transformer and the text tower over several GPUs) and bench_allreduce
	if(ENGINE_BACKEND STREQUAL "CUDA")
		find_library(NCCL_LIBRARY nccl)
	endif()
	if(NCCL_LIBRARY)
		target_compile_definitions(engine_kernels PUBLIC ENGINE_HAVE_NCCL=1)
		target_link_libraries(engine_kernels PUBLIC ${NCCL_LIBRARY})
		add_executable(bench_allreduce tests/3_bench/bench_allreduce.cpp)
		target_link_libraries(bench_allreduce PRIVATE engine_kernels ${NCCL_LIBRARY})
		set_target_properties(bench_allreduce PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	endif()
endif()
