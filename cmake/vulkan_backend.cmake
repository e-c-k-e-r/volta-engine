# vulkan_backend: headless Vulkan compute layer (src/backend/vulkan) and the SPIR-V parity test.
# ENGINE_BACKEND_VULKAN is defined for these targets only; the CUDA build is untouched.
find_package(Vulkan QUIET)
if(Vulkan_FOUND AND TARGET slang_kernels)
	# vk_*.cpp: device layer only; mod_*.cpp / stubs_*.cpp are the module layer of the VULKAN configuration (below)
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
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/flash.slang
			COMMENT "slangc ${entry} -> spirv")
		list(APPEND VK_FLASH_SPV ${sout})
	endforeach()
	# GLM kernels (src/kernels/slang/glm.slang: MLA latent, DSA scoring, KDA)
	foreach(entry latent_absorb_q latent_expand_out latent_store latent_attend latent_diag_scores dsa_score kda_gates_small kda_gates_big kda_step router_partials moe_combine_parts)
		set(sout ${SLANG_OUT}/${entry}.spv)
		add_custom_command(OUTPUT ${sout}
			COMMAND ${SLANGC} ${PROJECT_SOURCE_DIR}/src/kernels/slang/glm.slang -target spirv -stage compute -entry ${entry} -DENGINE_TARGET_CPU=0 -o ${sout}
			DEPENDS ${PROJECT_SOURCE_DIR}/src/kernels/slang/glm.slang
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
	add_custom_target(flash_spv DEPENDS ${VK_FLASH_SPV})
	add_dependencies(slang_kernels flash_spv)

	file(GLOB ENGINE_VK_MODULE_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/backend/vulkan/mod_*.cpp ${PROJECT_SOURCE_DIR}/src/backend/vulkan/stubs_*.cpp)
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
