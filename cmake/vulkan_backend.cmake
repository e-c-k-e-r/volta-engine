# vulkan_backend: headless Vulkan compute layer (src/backend/vulkan) and the SPIR-V parity test.
# ENGINE_BACKEND_VULKAN is defined for these targets only; the CUDA build is untouched.
find_package(Vulkan QUIET)
if(Vulkan_FOUND AND TARGET slang_kernels)
	file(GLOB VULKAN_BACKEND_SOURCES ${PROJECT_SOURCE_DIR}/src/backend/vulkan/*.cpp)
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
