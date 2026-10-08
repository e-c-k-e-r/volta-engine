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
	foreach(t test_ops_linear test_ops_attention)
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
