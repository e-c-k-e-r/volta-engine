# vulkan_backend: headless Vulkan compute layer (src/backend/vulkan) and the SPIR-V parity test.
# ENGINE_BACKEND_VULKAN is defined for these targets only; the CUDA build is untouched.
find_package(Vulkan QUIET)
if(Vulkan_FOUND AND TARGET slang_kernels)
	# vk_*.cpp: device layer only; mod_*.cpp / stubs_*.cpp (plus src/backend/stubs, shared with the CPU build) are the module layer of the VULKAN configuration (below)
	file(GLOB VULKAN_BACKEND_SOURCES ${PROJECT_SOURCE_DIR}/src/backend/vulkan/vk_*.cpp)
	add_library(engine_backend_vulkan STATIC ${VULKAN_BACKEND_SOURCES})
	# The main CMakeLists adds ENGINE_BACKEND_CUDA=1 globally; -U after the -D flags removes it for these targets
	target_compile_definitions(engine_backend_vulkan PUBLIC ENGINE_BACKEND_VULKAN=1)
	target_compile_options(engine_backend_vulkan PUBLIC -UENGINE_BACKEND_CUDA)
	target_include_directories(engine_backend_vulkan PUBLIC ${PROJECT_SOURCE_DIR}/inc)
	target_link_libraries(engine_backend_vulkan PUBLIC Vulkan::Vulkan)
	target_compile_features(engine_backend_vulkan PUBLIC cxx_std_17)
	set_target_properties(engine_backend_vulkan PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" POSITION_INDEPENDENT_CODE ON)

	add_executable(test_vulkan_kernels ${PROJECT_SOURCE_DIR}/tests/2_parity/test_vulkan_kernels.cpp)
	target_link_libraries(test_vulkan_kernels PRIVATE engine_backend_vulkan)
	target_compile_definitions(test_vulkan_kernels PRIVATE SLANG_SPV_DIR="${PROJECT_BINARY_DIR}/slang")
	add_dependencies(test_vulkan_kernels slang_kernels)
	set_target_properties(test_vulkan_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
else()
	message(STATUS "Vulkan or slang_kernels not found: Vulkan backend skipped")
endif()

# ---- VULKAN configuration (ENGINE_BACKEND=VULKAN): module layer over the SPIR-V kernels, no CUDA language
# OBJECT libraries (not archives) so a strong definition elsewhere overrides the weak stubs.
if(ENGINE_BACKEND STREQUAL "VULKAN")
	if(NOT Vulkan_FOUND OR NOT TARGET slang_kernels)
		message(FATAL_ERROR "ENGINE_BACKEND=VULKAN needs the Vulkan SDK and slangc (SPIR-V kernels)")
	endif()
	# The reference kernels are plain Slang outputs: build their SPIR-V in this configuration too
	foreach(t ref_lin_slang ref_attn_slang)
		if(TARGET ${t})
			add_dependencies(slang_kernels ${t})
		endif()
	endforeach()

	# Vulkan-only Slang kernels (src/kernels/slang/flash.slang: Flash-Next gated residual, PLE, QSA, selection, sparse attention, router, MoE combine);
	# the CPU backend has plain C++ for these
	set(VK_FLASH_ENTRIES gr_init gr_norm gr_gate gr_up_mix gr_apply gr_sigmoid_gated gr_stream_norm gr_add_broadcast gr_gemm_f32 gr_gate_act gr_up_epilogue
		ple_gate ple_conv ple_ring qsa_prep qsa_pool_update qsa_score dsa_select sparse_attn router_softmax_topk moe_combine_routes)
	set(VK_FLASH_SPV "")
	foreach(entry ${VK_FLASH_ENTRIES})
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/flash.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/flash.slang ${SLANG_COMMON}
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# GLM kernels (src/kernels/slang/glm.slang: MLA latent, DSA scoring, KDA)
	foreach(entry latent_absorb_q latent_expand_out latent_store latent_attend latent_diag_scores dsa_score kda_gates_small kda_gates_big kda_step router_partials moe_combine_parts)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/glm.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/glm.slang ${SLANG_COMMON}
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# Vision tower operators (src/kernels/slang/vision.slang)
	foreach(entry vision_f32_to_f16 vision_pos_interp vision_patch_embed vision_norm vision_qkv_rope vision_merge_gather vision_attention vision_bias_act vision_bias_residual)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/vision.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/vision.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# NVFP4 W4A16 linear (src/kernels/slang/nvfp4.slang)
	add_custom_command(OUTPUT ${SLANG_OUT}/nvfp4_gemm.spv
		COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/nvfp4.slang -target spirv -stage compute -entry nvfp4_gemm -DENGINE_TARGET_CPU=0 -o ${SLANG_OUT}/nvfp4_gemm.spv
		DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/nvfp4.slang
		COMMENT "slangc nvfp4_gemm -> spirv")
	list(APPEND VK_FLASH_SPV ${SLANG_OUT}/nvfp4_gemm.spv)
	# GGUF block weights (src/kernels/slang/gguf.slang)
	foreach(entry gguf_gemv_sg gguf_quantize_q8 gguf_gemm_tile gguf_dequant gguf_permute_rows gguf_quantize_q8_1x4 moe_local_ids moe_row_map)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/gguf.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/gguf.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# the head's top-k and log-sum-exp (src/kernels/slang/spec_topk.slang)
	foreach(entry topk_shard lse_shard topk_merge)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/spec_topk.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/spec_topk.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# the integer GEMV once per block type (gguf.slang GGUF_T)
	foreach(t 2 3 6 7 8 10 11 12 13 14 16 17 18 19 20 21 22 23 29)
		set(sout ${SLANG_OUT}/gguf_gemv_dot_${t}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/gguf.slang -target spirv -stage compute -entry gguf_gemv_dot -DENGINE_TARGET_CPU=0 -DGGUF_T=${t} ${GGUF_DOT_FLAGS} -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/gguf.slang
			COMMENT "slangc gguf_gemv_dot (type ${t}) -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# llama.cpp's mat-vec shaders (src/kernels/vulkan/ggml, GLSL) once per block type: FP16 activations and output, subgroup reduction
	find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})
	if(GLSLC)
		set(GGML_VK_DIR ${PROJECT_SOURCE_DIR}/src/kernels/vulkan/ggml)
		file(GLOB GGML_VK_DEPS ${GGML_VK_DIR}/*.glsl ${GGML_VK_DIR}/*.comp)
		foreach(pair 2:q4_0 3:q4_1 6:q5_0 7:q5_1 8:q8_0 20:iq4_nl 10:q2_k 11:q3_k 12:q4_k 13:q5_k 14:q6_k 16:iq2_xxs 17:iq2_xs 18:iq3_xxs 19:iq1_s 21:iq3_s 22:iq2_s 23:iq4_xs 29:iq1_m)
			string(REPLACE ":" ";" pl ${pair})
			list(GET pl 0 id)
			list(GET pl 1 tname)
			string(TOUPPER ${tname} TU)
			set(src ${GGML_VK_DIR}/mul_mat_vec_${tname}.comp)
			if(NOT EXISTS ${src})
				set(src ${GGML_VK_DIR}/mul_mat_vec.comp)
			endif()
			set(sout ${SLANG_OUT}/ggml_mmv_${id}.spv)
			add_custom_command(OUTPUT ${sout}
				COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -DDATA_A_${TU}=1 -DB_TYPE=float16_t -DB_TYPEV2=f16vec2 -DB_TYPEV4=f16vec4
					-DD_TYPE=float16_t -DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2 -DUSE_SUBGROUP_ADD_NO_SHMEM=1 ${src} -o ${sout}
				DEPENDS ${GGML_VK_DEPS}
				COMMENT "glslc ggml mul_mat_vec ${tname} -> spirv")
			list(APPEND VK_FLASH_SPV ${sout})
			# and its expert-indexed form (MUL_MAT_ID: the routed experts of one token, the expert ids from a buffer)
			set(sout ${SLANG_OUT}/ggml_mmvid_${id}.spv)
			add_custom_command(OUTPUT ${sout}
				COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -DMUL_MAT_ID=1 -DDATA_A_${TU}=1 -DB_TYPE=float16_t -DB_TYPEV2=f16vec2
					-DB_TYPEV4=f16vec4 -DD_TYPE=float16_t -DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2 -DUSE_SUBGROUP_ADD_NO_SHMEM=1 ${src} -o ${sout}
				DEPENDS ${GGML_VK_DEPS}
				COMMENT "glslc ggml mul_mat_vec_id ${tname} -> spirv")
			list(APPEND VK_FLASH_SPV ${sout})
		endforeach()
		# and its int8 matmul (q8_1 activations) for the types that have one, as llama.cpp's FP16 build of it, FP16 output
		foreach(pair 2:q4_0 3:q4_1 6:q5_0 7:q5_1 8:q8_0 10:q2_k 11:q3_k 12:q4_k 13:q5_k 14:q6_k 21:iq3_s 23:iq4_xs)
			string(REPLACE ":" ";" pl ${pair})
			list(GET pl 0 id)
			list(GET pl 1 tname)
			string(TOUPPER ${tname} TU)
			set(sout ${SLANG_OUT}/ggml_mmq_${id}.spv)
			add_custom_command(OUTPUT ${sout}
				COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -DFLOAT16=1 -DACC_TYPE=float -DACC_TYPEV2=vec2 -DACC_TYPE_VEC4=vec4
					-DFLOAT_TYPE=float16_t -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DFLOAT_TYPEV8=f16mat2x4 -DDATA_A_${TU}=1 -DD_TYPE=float16_t
					${GGML_VK_DIR}/mul_mmq.comp -o ${sout}
				DEPENDS ${GGML_VK_DEPS}
				COMMENT "glslc ggml mul_mmq ${tname} -> spirv")
			list(APPEND VK_FLASH_SPV ${sout})
			set(sout ${SLANG_OUT}/ggml_mmqid_${id}.spv)
			add_custom_command(OUTPUT ${sout}
				COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -DMUL_MAT_ID=1 -DFLOAT16=1 -DACC_TYPE=float -DACC_TYPEV2=vec2 -DACC_TYPE_VEC4=vec4
					-DFLOAT_TYPE=float16_t -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DFLOAT_TYPEV8=f16mat2x4 -DDATA_A_${TU}=1 -DD_TYPE=float16_t
					${GGML_VK_DIR}/mul_mmq.comp -o ${sout}
				DEPENDS ${GGML_VK_DEPS}
				COMMENT "glslc ggml mul_mmq_id ${tname} -> spirv")
			list(APPEND VK_FLASH_SPV ${sout})
		endforeach()
		# and its matmul with the weight dequantized as it loads (lookup-table types without an int8 matmul): FP16 activations, fp32 sums
		foreach(pair 16:iq2_xxs:8 17:iq2_xs:8 18:iq3_xxs:4 19:iq1_s:8 20:iq4_nl:4 22:iq2_s:8 29:iq1_m:8)
			string(REPLACE ":" ";" pl ${pair})
			list(GET pl 0 id)
			list(GET pl 1 tname)
			list(GET pl 2 lva)
			string(TOUPPER ${tname} TU)
			set(sout ${SLANG_OUT}/ggml_mm_${id}.spv)
			add_custom_command(OUTPUT ${sout}
				COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -DFLOAT16=1 -DACC_TYPE=float -DACC_TYPEV2=vec2 -DACC_TYPE_VEC4=vec4
					-DFLOAT_TYPE=float16_t -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DFLOAT_TYPEV8=f16mat2x4 -DDATA_A_${TU}=1 -DLOAD_VEC_A=${lva}
					-DLOAD_VEC_B=8 -DB_TYPE=f16mat2x4 -DB_TYPE_SCALAR=float16_t -DB_TYPEV4=f16vec4 -DD_TYPE=float16_t ${GGML_VK_DIR}/mul_mm.comp -o ${sout}
				DEPENDS ${GGML_VK_DEPS}
				COMMENT "glslc ggml mul_mm ${tname} -> spirv")
			list(APPEND VK_FLASH_SPV ${sout})
			set(sout ${SLANG_OUT}/ggml_mmid_${id}.spv)
			add_custom_command(OUTPUT ${sout}
				COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -DMUL_MAT_ID=1 -DFLOAT16=1 -DACC_TYPE=float -DACC_TYPEV2=vec2 -DACC_TYPE_VEC4=vec4
					-DFLOAT_TYPE=float16_t -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DFLOAT_TYPEV8=f16mat2x4 -DDATA_A_${TU}=1 -DLOAD_VEC_A=${lva}
					-DLOAD_VEC_B=8 -DB_TYPE=f16mat2x4 -DB_TYPE_SCALAR=float16_t -DB_TYPEV4=f16vec4 -DD_TYPE=float16_t ${GGML_VK_DIR}/mul_mm.comp -o ${sout}
				DEPENDS ${GGML_VK_DEPS}
				COMMENT "glslc ggml mul_mm_id ${tname} -> spirv")
			list(APPEND VK_FLASH_SPV ${sout})
		endforeach()
	else()
		message(STATUS "glslc not found: the Vulkan GGUF projections run on the Slang kernels alone")
	endif()
	add_custom_target(flash_spv DEPENDS ${VK_FLASH_SPV})
	add_dependencies(slang_kernels flash_spv)

	file(GLOB ENGINE_VK_MODULE_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/backend/vulkan/mod_*.cpp ${PROJECT_SOURCE_DIR}/src/backend/vulkan/stubs_*.cpp ${PROJECT_SOURCE_DIR}/src/backend/stubs/*.cpp)
	add_library(engine_vk_host OBJECT ${ENGINE_HOST_SOURCES})
	add_library(engine_vk_modules OBJECT ${ENGINE_VK_MODULE_SOURCES})
	target_link_libraries(engine_vk_modules PUBLIC engine_backend_vulkan)
	target_compile_definitions(engine_vk_modules PRIVATE SLANG_SPV_DIR="${PROJECT_BINARY_DIR}/slang")
	set(ENGINE_VK_OBJECTS $<TARGET_OBJECTS:engine_vk_host> $<TARGET_OBJECTS:engine_vk_modules>)

	add_executable(test_vulkan_smoke tests/2_parity/test_vulkan_smoke.cpp $<TARGET_OBJECTS:engine_vk_modules>)
	target_link_libraries(test_vulkan_smoke PRIVATE engine_backend_vulkan)
	add_dependencies(test_vulkan_smoke slang_kernels)

	add_executable(test_vulkan_ref tests/2_parity/test_vulkan_ref.cpp)
	target_link_libraries(test_vulkan_ref PRIVATE engine_backend_vulkan)
	target_compile_definitions(test_vulkan_ref PRIVATE SLANG_SPV_DIR="${PROJECT_BINARY_DIR}/slang")
	add_dependencies(test_vulkan_ref slang_kernels)
	set_target_properties(test_vulkan_smoke test_vulkan_ref PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	# Multi-device collective through the host (one thread per rank, each on its own device)
	add_executable(test_vulkan_comm tests/2_parity/test_vulkan_comm.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(test_vulkan_comm PRIVATE engine_backend_vulkan pthread)
	add_dependencies(test_vulkan_comm slang_kernels)
	set_target_properties(test_vulkan_comm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	add_executable(test_mtp tests/4_coherence/test_mtp.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(test_mtp PRIVATE engine_backend_vulkan pthread)
	set_target_properties(test_mtp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	add_dependencies(test_mtp slang_kernels)

	# GGUF block types against the ggml fixtures (tests/tools/gguf_fixtures.py), same source as the CUDA and CPU builds
	add_executable(bench_gguf_shapes tests/3_bench/bench_gguf_shapes.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(bench_gguf_shapes PRIVATE engine_backend_vulkan pthread)
	add_dependencies(bench_gguf_shapes slang_kernels)
	set_target_properties(bench_gguf_shapes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	add_executable(test_gguf_linear tests/2_parity/test_gguf_linear.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(test_gguf_linear PRIVATE engine_backend_vulkan pthread)
	add_dependencies(test_gguf_linear slang_kernels)
	set_target_properties(test_gguf_linear PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	# End-to-end single-rank forward (--check against the CUDA --dump golden), same source as test_cpu_forward
	add_executable(test_vulkan_forward tests/2_parity/test_cpu_forward.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(test_vulkan_forward PRIVATE engine_backend_vulkan pthread)
	add_dependencies(test_vulkan_forward slang_kernels)
	set_target_properties(test_vulkan_forward PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	# Op-level goldens through the module API (--check against the CUDA --dump)
	foreach(t test_ops_linear test_ops_attention test_ops_moe test_ops_flash test_ops_vision)
		add_executable(${t} tests/2_parity/${t}.cpp ${ENGINE_VK_OBJECTS})
		target_link_libraries(${t} PRIVATE engine_backend_vulkan pthread)
		add_dependencies(${t} slang_kernels)
		set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	endforeach()

	# The server entry point, to show how far the host sources link (the quantized linears and attention are stubs)
	add_executable(volta_server src/server/main.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(volta_server PRIVATE engine_backend_vulkan pthread)
	set_target_properties(volta_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()
