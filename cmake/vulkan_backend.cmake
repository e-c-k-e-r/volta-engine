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
	set(VK_FLASH_ENTRIES gr_init gr_norm gr_gate gr_up_mix gr_apply gr_sigmoid_gated gr_stream_norm gr_add_broadcast gr_gemm_f32 gr_split_sum gr_gate_act gr_up_epilogue
		ple_gate ple_conv ple_ring qsa_prep qsa_pool_update qsa_score dsa_select sparse_attn router_sm_partials router_sm_select moe_combine_routes)
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
	# The attention's input step (src/kernels/slang/attn_prep.slang: head norms, rotary embedding and the FP16 cache store in one kernel)
	foreach(entry norm_rope_store)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/attn_prep.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/attn_prep.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# The chunked delta rule of the GDN prefill (src/kernels/slang/gdn_chunk.slang)
	foreach(entry gdn_chunk_prep gdn_chunk_scan gdn_chunk_out)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/gdn_chunk.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/gdn_chunk.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# The chunked delta rule of the KDA prefill (src/kernels/slang/kda_chunk.slang)
	foreach(entry kda_chunk_prep kda_chunk_scan kda_chunk_out kda_chunk_norm)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/kda_chunk.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/kda_chunk.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# Vision tower operators (src/kernels/slang/vision.slang)
	foreach(entry vision_f32_to_f16 vision_pos_interp vision_patch_embed vision_norm vision_qkv_rope vision_merge_gather vision_attention vision_bias_act vision_bias_residual vision_qkv_rope2d vision_pool_std)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/vision.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/vision.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# EXL3 trellis linear (src/kernels/slang/trellis.slang): the rotation once, the GEMV and the dequantization once per width code
	foreach(entry trellis_had128 trellis_split_sum)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/trellis.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/trellis.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	foreach(code 1 2 3 4 5 6 7 8 9 10 11)
		# trellis_gemv (up to 8 rows: 2 tiles a step), trellis_gemv1 (one row: 4 tiles a step), trellis_gemv16 (up to 16 rows), trellis_dequant;
		# each also with the codebook's byte sum as a packed dot product (_dot, devices with shaderIntegerDotProduct)
		foreach(variant trellis_gemv:trellis_gemv:8:2 trellis_gemv1:trellis_gemv:1:4 trellis_gemv16:trellis_gemv:16:1 trellis_dequant:trellis_dequant:8:1)
			string(REPLACE ":" ";" vparts ${variant})
			list(GET vparts 0 vname)
			list(GET vparts 1 entry)
			list(GET vparts 2 vrows)
			list(GET vparts 3 vunroll)
			foreach(dot 0 1)
				# and the extended trellis' forms (_ext: tile starts / gains, TR_EXT)
				foreach(ext 0 1)
					set(suffix "")
					if(dot)
						set(suffix "_dot")
					endif()
					if(ext)
						set(suffix "${suffix}_ext")
					endif()
					set(sout ${SLANG_OUT}/${vname}_c${code}${suffix}.spv)
					add_custom_command(OUTPUT ${sout}
						COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/trellis.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -DTR_CODE=${code}
							-DTR_GEMV_ROWS=${vrows} -DTR_UNROLL=${vunroll} -DTR_THREADS=256 -DTR_DOT=${dot} -DTR_EXT=${ext} -o ${sout}
						DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/trellis.slang
						COMMENT "slangc ${vname} (code ${code}${suffix}) -> spirv")
					list(APPEND VK_FLASH_SPV ${sout})
				endforeach()
			endforeach()
		endforeach()
	endforeach()
	# vall-e TTS and NeMo codec operators (src/kernels/slang/tts.slang)
	foreach(entry tts_digits_values tts_embed_mix tts_add_norm tts_add_bias tts_qk_norm_rope tts_heads_gate tts_swiglu tts_gather_rows tts_bias_act tts_cond_silu
		tts_ln_mod tts_gate_add tts_replicate tts_guide_step tts_snap codec_dequant codec_quant codec_act codec_cols codec_up_cols codec_interleave codec_channel_bias
		codec_axpy codec_clamp)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/tts.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/tts.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# Tiled attention (src/kernels/slang/fa.slang)
	foreach(entry fa_attn fa_decode_256 fa_decode_512 fa_reduce fa_finish fa_dense_64)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/fa.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/fa.slang ${SLANG_COMMON}
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# NVFP4 W4A16 linear and the NVFP4 experts (src/kernels/slang/nvfp4.slang)
	foreach(entry nvfp4_gemm nvfp4_gemv nvfp4_dequant nvfp4_moe_gate_up nvfp4_moe_down nvfp4_moe_gate_up_grp nvfp4_moe_down_grp)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/nvfp4.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/nvfp4.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# nvfp4_gemv also for 1, 2 and 4 rows (nvfp4_gemv1 / 2 / 4; nvfp4_gemv is the 8-row form)
	foreach(vrows 1 2 4)
		set(sout ${SLANG_OUT}/nvfp4_gemv${vrows}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/nvfp4.slang -target spirv -stage compute -entry nvfp4_gemv -DENGINE_TARGET_CPU=0 -DNVG_ROWS=${vrows} -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/nvfp4.slang
			COMMENT "slangc nvfp4_gemv (${vrows} rows) -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
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
		# the multi-device collective's sum (src/kernels/slang/comm.slang)
	foreach(entry ar_sum)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/comm.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/comm.slang
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
	# llama.cpp's mat-vec shaders (src/kernels/vulkan/ggml, GLSL) once per block type (and FP16 weights): FP16 activations and output, subgroup reduction
	find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})
	if(GLSLC)
		set(GGML_VK_DIR ${PROJECT_SOURCE_DIR}/src/kernels/vulkan/ggml)
		file(GLOB GGML_VK_DEPS ${GGML_VK_DIR}/*.glsl ${GGML_VK_DIR}/*.comp)
		foreach(pair 1:f16 2:q4_0 3:q4_1 6:q5_0 7:q5_1 8:q8_0 20:iq4_nl 10:q2_k 11:q3_k 12:q4_k 13:q5_k 14:q6_k 16:iq2_xxs 17:iq2_xs 18:iq3_xxs 19:iq1_s 21:iq3_s 22:iq2_s 23:iq4_xs 29:iq1_m)
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
		# and its matmul with the weight dequantized as it loads (lookup-table types without an int8 matmul, FP16 weights): FP16 activations, fp32 sums
		foreach(pair 1:f16:8 16:iq2_xxs:8 17:iq2_xs:8 18:iq3_xxs:4 19:iq1_s:8 20:iq4_nl:4 22:iq2_s:8 29:iq1_m:8)
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
		# the FP16 matmul with FP32 output (GatedResidual::gemm_f32)
		set(sout ${SLANG_OUT}/ggml_mm_1_f32.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -DFLOAT16=1 -DACC_TYPE=float -DACC_TYPEV2=vec2 -DACC_TYPE_VEC4=vec4
				-DFLOAT_TYPE=float16_t -DFLOAT_TYPEV2=f16vec2 -DFLOAT_TYPEV4=f16vec4 -DFLOAT_TYPEV8=f16mat2x4 -DDATA_A_F16=1 -DLOAD_VEC_A=8
				-DLOAD_VEC_B=8 -DB_TYPE=f16mat2x4 -DB_TYPE_SCALAR=float16_t -DB_TYPEV4=f16vec4 -DD_TYPE=float ${GGML_VK_DIR}/mul_mm.comp -o ${sout}
			DEPENDS ${GGML_VK_DEPS}
			COMMENT "glslc ggml mul_mm f16 -> f32 -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
		# the FP32 matmul (the TTS codec's convolutions): vec4 loads where ALIGNED, scalar ones with K bounds otherwise
		set(sout ${SLANG_OUT}/ggml_mm_f32.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${GLSLC} -fshader-stage=compute --target-env=vulkan1.2 -O -DACC_TYPE=float -DACC_TYPEV2=vec2 -DACC_TYPE_VEC4=vec4
				-DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2 -DFLOAT_TYPEV4=vec4 -DFLOAT_TYPEV8=mat2x4 -DDATA_A_F32=1 -DLOAD_VEC_A=4
				-DLOAD_VEC_B=4 -DB_TYPE=vec4 -DB_TYPE_SCALAR=float -DB_TYPEV4=vec4 -DD_TYPE=float ${GGML_VK_DIR}/mul_mm.comp -o ${sout}
			DEPENDS ${GGML_VK_DEPS}
			COMMENT "glslc ggml mul_mm f32 -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	else()
		message(STATUS "glslc not found: the Vulkan GGUF projections run on the Slang kernels alone")
	endif()
	# The CUDA backend's FP16 tensor-core GEMM and NVFP4 linear kernels as PTX for the launches through VK_NV_cuda_kernel_launch (NVIDIA GPUs
	# without cooperative matrices: Volta; src/backend/vulkan/mod_cuda.cpp), where nvcc is found
	find_program(ENGINE_NVCC nvcc HINTS /opt/cuda/bin /usr/local/cuda/bin ENV CUDA_PATH PATH_SUFFIXES bin)
	if(ENGINE_NVCC)
		set(KDIR ${PROJECT_SOURCE_DIR}/src/kernels/cuda)
		foreach(ptx "cuda_gemm|gemm/gemm_f16_vk.cu|gemm/gemm_f16_kernel.cuh" "cuda_nvfp4|nvfp4/nvfp4_vk.cu|nvfp4/nvfp4_linear_kernels.cuh;nvfp4/nvfp4_common.cuh;nvfp4/nvfp4_frag.h;mma_sm80.cuh;mma_frag.h")
			string(REPLACE "|" ";" parts "${ptx}")
			list(GET parts 0 name)
			list(GET parts 1 src)
			list(SUBLIST parts 2 -1 deps)
			list(TRANSFORM deps PREPEND ${KDIR}/)
			set(sout ${SLANG_OUT}/${name}.ptx)
			add_custom_command(OUTPUT ${sout}
				COMMAND ${ENGINE_NVCC} -arch=compute_70 -ptx -std=c++17 -O3 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets -diag-suppress 177
					${KDIR}/${src} -o ${sout}
				DEPENDS ${KDIR}/${src} ${deps}
				COMMENT "nvcc ${src} -> ptx")
			list(APPEND VK_FLASH_SPV ${sout})
		endforeach()
	else()
		message(STATUS "nvcc not found: on NVIDIA GPUs without cooperative matrices the Vulkan GEMMs and NVFP4 linears stay on the shader cores")
	endif()
	add_custom_target(flash_spv DEPENDS ${VK_FLASH_SPV})
	add_dependencies(slang_kernels flash_spv)

	file(GLOB ENGINE_VK_MODULE_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/backend/vulkan/mod_*.cpp ${PROJECT_SOURCE_DIR}/src/backend/vulkan/stubs_*.cpp ${PROJECT_SOURCE_DIR}/src/backend/stubs/*.cpp)
	# the host sources, the TTS (its text front-end and endpoint with propan2ol, which every target linking these objects then links) and the
	# dense-weight uploader that converts on the host
	if(ENGINE_TTS)
		add_compile_definitions(ENGINE_TTS=1)
		target_link_libraries(engine_backend_vulkan PUBLIC propan2ol::propan2ol)
	endif()
	add_library(engine_vk_host OBJECT ${ENGINE_HOST_SOURCES} ${ENGINE_TTS_SOURCES} ${PROJECT_SOURCE_DIR}/src/loader/device_upload_host.cpp)
	if(ENGINE_TTS)
		target_link_libraries(engine_vk_host PRIVATE propan2ol::propan2ol)
	endif()
	add_library(engine_vk_modules OBJECT ${ENGINE_VK_MODULE_SOURCES})
	target_link_libraries(engine_vk_modules PUBLIC engine_backend_vulkan)
	target_compile_definitions(engine_vk_modules PRIVATE SLANG_SPV_DIR="${PROJECT_BINARY_DIR}/slang")
	set(ENGINE_VK_OBJECTS $<TARGET_OBJECTS:engine_vk_host> $<TARGET_OBJECTS:engine_vk_modules>)

	# the module objects without the host ones, but for the GGUF block table their stubs size weights with
	add_executable(test_vulkan_smoke tests/2_parity/test_vulkan_smoke.cpp src/loader/gguf_types.cpp $<TARGET_OBJECTS:engine_vk_modules>)
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

	# The tiled attention against the reference kernel on the same device
	add_executable(test_vulkan_attention tests/2_parity/test_vulkan_attention.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(test_vulkan_attention PRIVATE engine_backend_vulkan pthread)
	add_dependencies(test_vulkan_attention slang_kernels)
	set_target_properties(test_vulkan_attention PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	# Gemma 4 vision tower against the exllamav3 golden (same source as the CUDA build's)
	add_executable(test_gemma4_vision tests/2_parity/test_gemma4_vision.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(test_gemma4_vision PRIVATE engine_backend_vulkan pthread)
	add_dependencies(test_gemma4_vision slang_kernels)
	set_target_properties(test_gemma4_vision PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	# Op-level goldens through the module API (--check against the CUDA --dump)
	foreach(t test_ops_linear test_ops_attention test_ops_moe test_ops_flash test_ops_vision)
		add_executable(${t} tests/2_parity/${t}.cpp ${ENGINE_VK_OBJECTS})
		target_link_libraries(${t} PRIVATE engine_backend_vulkan pthread)
		add_dependencies(${t} slang_kernels)
		set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	endforeach()

	# vall-e TTS from the command line and its golden checks (same source as the CUDA build's)
	add_executable(tts_generate src/tts/tts_generate_main.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(tts_generate PRIVATE engine_backend_vulkan pthread)
	add_dependencies(tts_generate slang_kernels)
	add_executable(bench_tts_ops tests/3_bench/bench_tts_ops.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(bench_tts_ops PRIVATE engine_backend_vulkan pthread)
	add_dependencies(bench_tts_ops slang_kernels)
	add_executable(bench_trellis_shapes tests/3_bench/bench_trellis_shapes.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(bench_trellis_shapes PRIVATE engine_backend_vulkan pthread)
	add_dependencies(bench_trellis_shapes slang_kernels)
	set_target_properties(tts_generate bench_tts_ops bench_trellis_shapes PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	if(ENGINE_TTS)
		# the text front-end against vall_e's phonemes (host only)
		add_executable(test_tts_g2p tests/2_parity/test_tts_g2p.cpp ${ENGINE_VK_OBJECTS})
		target_link_libraries(test_tts_g2p PRIVATE engine_backend_vulkan pthread)
		set_target_properties(test_tts_g2p PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	endif()

	# The server entry point, to show how far the host sources link (the quantized linears and attention are stubs)
	add_executable(volta_server src/server/main.cpp ${ENGINE_VK_OBJECTS})
	target_link_libraries(volta_server PRIVATE engine_backend_vulkan pthread)
	set_target_properties(volta_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	# libvolta (ENGINE_LIBRARY): inc/volta/volta.h over the host and module objects as a shared library, every kernel's SPIR-V linked in (a
	# generated table of .incbin blobs, looked up by name before the files; ENGINE_SPV_DIR still points the loader at a directory) and only
	# namespace volta exported (src/library/volta.map); the objects it takes are position-independent with hidden symbols
	if(ENGINE_LIBRARY)
		set(VOLTA_SPV ${VK_FLASH_SPV})
		foreach(f ${SLANG_GENERATED} ${REF_ATTN_GENERATED} ${REF_LIN_GENERATED})
			if(f MATCHES "\\.spv$")
				list(APPEND VOLTA_SPV ${f})
			endif()
		endforeach()
		list(REMOVE_DUPLICATES VOLTA_SPV)
		set(asm "")
		set(decl "")
		set(table "")
		set(i 0)
		foreach(f ${VOLTA_SPV})
			get_filename_component(n ${f} NAME_WLE)
			string(APPEND asm [=[	".balign 4\nvolta_spv_]=] "${i}" [=[:\n.incbin \"]=] "${f}" [=[\"\nvolta_spv_]=] "${i}" [=[_end:\n"]=] "\n")
			string(APPEND decl "extern \"C\" const uint8_t volta_spv_${i}[], volta_spv_${i}_end[];\n")
			string(APPEND table "\t{ \"${n}\", volta_spv_${i}, volta_spv_${i}_end },\n")
			math(EXPR i "${i} + 1")
		endforeach()
		set(VOLTA_SPV_CPP ${PROJECT_BINARY_DIR}/volta_spv.cpp)
		file(WRITE ${VOLTA_SPV_CPP}.new "// Generated by cmake/vulkan_backend.cmake (ENGINE_LIBRARY): the kernels' SPIR-V linked into libvolta
#include <cstddef>
#include <cstdint>
#include <cstring>

__asm__(
	\".section .rodata\\n\"
${asm}	\".previous\\n\" );

${decl}
namespace engine::backend::vk {

namespace {
struct Spv { const char* name; const uint8_t* begin; const uint8_t* end; };
const Spv kSpv[] = {
${table}};
} // namespace

const uint8_t* embedded_spv( const char* name, size_t* bytes ) {
	for ( const Spv& s : kSpv ) {
		if ( std::strcmp( s.name, name ) == 0 ) {
			*bytes = static_cast<size_t>( s.end - s.begin );
			return s.begin;
		}
	}
	return nullptr;
}

} // namespace engine::backend::vk
")
		configure_file(${VOLTA_SPV_CPP}.new ${VOLTA_SPV_CPP} COPYONLY)
		set_source_files_properties(${VOLTA_SPV_CPP} PROPERTIES OBJECT_DEPENDS "${VOLTA_SPV}")
		add_library(volta SHARED ${PROJECT_SOURCE_DIR}/src/library/volta.cpp ${VOLTA_SPV_CPP} ${ENGINE_VK_OBJECTS})
		add_dependencies(volta slang_kernels)
		target_link_libraries(volta PRIVATE engine_backend_vulkan pthread)
		target_compile_definitions(volta PRIVATE VOLTA_BUILD=1)
		target_link_options(volta PRIVATE -Wl,--version-script=${PROJECT_SOURCE_DIR}/src/library/volta.map -Wl,--exclude-libs,ALL -Wl,--no-undefined)
		set_target_properties(volta PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}" LINK_DEPENDS ${PROJECT_SOURCE_DIR}/src/library/volta.map)
		set_target_properties(engine_vk_host engine_vk_modules engine_backend_vulkan volta PROPERTIES POSITION_INDEPENDENT_CODE ON
			CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
		# the library from a host's side (volta/volta.h and libvolta.so only; --adopt hands it the test's own Vulkan device)
		add_executable(test_library ${PROJECT_SOURCE_DIR}/tests/4_coherence/test_library.cpp)
		target_link_libraries(test_library PRIVATE volta Vulkan::Vulkan pthread)
		set_target_properties(test_library PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
	endif()
endif()
