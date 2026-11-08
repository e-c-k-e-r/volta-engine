# cpu_backend: CPU computing backend (ENGINE_BACKEND=CPU). Included by the main CMakeLists.txt after the Slang block.
# OBJECT libraries (not archives) so a strong definition from another lane overrides the weak stubs in src/backend/cpu.
if(NOT ENGINE_BACKEND STREQUAL "CPU")
	return()
endif()

if(NOT SLANGC)
	message(FATAL_ERROR "ENGINE_BACKEND=CPU needs slangc (the CPU kernels are the Slang host target)")
endif()

# Host sources that compile without the vendor runtime (see the report in the commit message for what blocks the rest)
set(ENGINE_CPU_HOST_SOURCES ${ENGINE_HOST_SOURCES})

file(GLOB ENGINE_CPU_BACKEND_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/backend/cpu/*.cpp ${PROJECT_SOURCE_DIR}/src/backend/stubs/*.cpp)

add_library(engine_cpu_host OBJECT ${ENGINE_CPU_HOST_SOURCES})

add_library(engine_cpu_backend OBJECT ${ENGINE_CPU_BACKEND_SOURCES})
target_include_directories(engine_cpu_backend PRIVATE ${SLANG_INCLUDE_DIR})

# GGUF block kernels: llama.cpp's CPU quantization files (MIT, src/kernels/cpu/gguf/ggml), C, plus the shim for the ggml symbols they use
enable_language(C)
set(GGML_CPU_DIR ${PROJECT_SOURCE_DIR}/src/kernels/cpu/gguf/ggml)
add_library(engine_cpu_ggml OBJECT ${GGML_CPU_DIR}/ggml-quants.c ${GGML_CPU_DIR}/ggml-cpu/quants.c ${GGML_CPU_DIR}/ggml-cpu/arch/x86/quants.c
	${PROJECT_SOURCE_DIR}/src/kernels/cpu/gguf/ggml_shim.c ${PROJECT_SOURCE_DIR}/src/kernels/cpu/gguf/gguf_cpu.cpp)
target_include_directories(engine_cpu_ggml PRIVATE ${GGML_CPU_DIR} ${GGML_CPU_DIR}/ggml-cpu)
target_include_directories(engine_cpu_ggml PRIVATE ${PROJECT_SOURCE_DIR}/inc)
set_target_properties(engine_cpu_ggml PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
target_compile_options(engine_cpu_ggml PRIVATE -O3 $<$<BOOL:${ENGINE_CPU_NATIVE}>:-march=native> $<$<COMPILE_LANGUAGE:C>:-w>)

# Generated Slang host kernels (their own units: the prelude defines types that clash with nothing else, but keeps -Wall noise out)
add_library(engine_cpu_slang OBJECT ${SLANG_CPU_SOURCES})
add_dependencies(engine_cpu_slang slang_kernels)
target_include_directories(engine_cpu_slang PRIVATE ${SLANG_INCLUDE_DIR})

set(ENGINE_CPU_OBJECTS $<TARGET_OBJECTS:engine_cpu_host> $<TARGET_OBJECTS:engine_cpu_backend> $<TARGET_OBJECTS:engine_cpu_slang> $<TARGET_OBJECTS:engine_cpu_ggml>)

add_executable(test_cpu_smoke tests/2_parity/test_cpu_smoke.cpp ${ENGINE_CPU_OBJECTS})
set_target_properties(test_cpu_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_link_libraries(test_cpu_smoke PRIVATE pthread)

add_custom_target(run_cpu_smoke COMMAND test_cpu_smoke DEPENDS test_cpu_smoke)

# The production server entry point, to show how far the host sources link (it cannot serve a model yet: the quantized linears are stubs)
add_executable(volta_server src/server/main.cpp ${ENGINE_CPU_OBJECTS})
target_link_libraries(volta_server PRIVATE pthread)
set_target_properties(volta_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

add_executable(test_cpu_forward tests/2_parity/test_cpu_forward.cpp ${ENGINE_CPU_OBJECTS})
set_target_properties(test_cpu_forward PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_link_libraries(test_cpu_forward PRIVATE pthread)

add_executable(test_gguf_linear tests/2_parity/test_gguf_linear.cpp ${ENGINE_CPU_OBJECTS})
set_target_properties(test_gguf_linear PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_link_libraries(test_gguf_linear PRIVATE pthread)
