# reference_kernels: build fragment owned by one Wave 2 lane (included by the main CMakeLists.txt when present)

# ---- E2 begin
# Portable reference attention kernels (src/shaders/ref_attention.slang, ref_latent.slang): cuda / spirv / cpp outputs next to
# the other Slang kernels. The CUDA launchers (src/shaders/cuda/ref_attn*.cu, ref_latent*.cu) are picked up by the
# engine_slang_cuda glob, so that library only needs the ordering dependency on the generated sources.
if(SLANGC)
	set(REF_ATTN_KERNELS ref_attention:ref_attn_decode ref_latent:ref_latent_decode)
	set(REF_ATTN_GENERATED "")
	set(REF_ATTN_CPU_SOURCES "")
	foreach(item ${REF_ATTN_KERNELS})
		string(REPLACE ":" ";" parts ${item})
		list(GET parts 0 sfile)
		list(GET parts 1 entry)
		set(ssrc ${PROJECT_SOURCE_DIR}/src/shaders/${sfile}.slang)
		foreach(tgt cuda:cuda.cu:0 spirv:spv:0 cpp:cpp:1)
			string(REPLACE ":" ";" tparts ${tgt})
			list(GET tparts 0 target)
			list(GET tparts 1 ext)
			list(GET tparts 2 cpu)
			# The host entry carries a _cpu suffix: the CUDA launch stub of the same name would collide at link time
			set(eentry ${entry})
			if(cpu)
				set(eentry ${entry}_cpu)
			endif()
			set(sout ${SLANG_OUT}/${eentry}.${ext})
			add_custom_command(OUTPUT ${sout}
				COMMAND ${SLANGC} ${ssrc} -target ${target} -stage compute -entry ${eentry} -DENGINE_TARGET_CPU=${cpu} -o ${sout}
				DEPENDS ${ssrc}
				COMMENT "slangc ${entry} -> ${target}")
			list(APPEND REF_ATTN_GENERATED ${sout})
		endforeach()
		list(APPEND REF_ATTN_CPU_SOURCES ${SLANG_OUT}/${entry}_cpu.cpp)
	endforeach()
	add_custom_target(ref_attn_slang DEPENDS ${REF_ATTN_GENERATED})
	if(TARGET engine_slang_cuda)
		add_dependencies(engine_slang_cuda ref_attn_slang)
	endif()

	find_program(REF_ATTN_SPIRV_VAL spirv-val)
	if(REF_ATTN_SPIRV_VAL)
		add_custom_target(ref_attn_spirv_val
			COMMAND ${REF_ATTN_SPIRV_VAL} ${SLANG_OUT}/ref_attn_decode.spv
			COMMAND ${REF_ATTN_SPIRV_VAL} ${SLANG_OUT}/ref_latent_decode.spv
			DEPENDS ref_attn_slang)
	endif()

	# Host-target kernels are separate translation units (the Slang C++ prelude is not mixed with the CUDA headers)
	set_source_files_properties(${REF_ATTN_CPU_SOURCES} PROPERTIES GENERATED TRUE)
	add_library(ref_attn_cpu STATIC ${REF_ATTN_CPU_SOURCES})
	add_dependencies(ref_attn_cpu ref_attn_slang)
	target_include_directories(ref_attn_cpu PRIVATE ${SLANG_INCLUDE_DIR})
	set_target_properties(ref_attn_cpu PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	add_executable(test_ref_attention tests/2_parity/test_ref_attention.cpp)
	target_include_directories(test_ref_attention PRIVATE src/shaders/cuda)
	target_link_libraries(test_ref_attention PRIVATE ref_attn_cpu engine_kernels engine_slang_cuda)
	add_dependencies(test_ref_attention ref_attn_slang)
	set_target_properties(test_ref_attention PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()
# ---- E2 end

# ---- E1 begin
# Portable reference linear algebra (ref_gemm.slang, ref_trellis.slang): cuda / spirv / cpp outputs, CUDA launcher library, host + GPU test
find_program(SLANGC slangc HINTS /opt/shader-slang/bin)
if(SLANGC)
	get_filename_component(REF_LIN_SLANG_BIN_DIR ${SLANGC} DIRECTORY)
	set(REF_LIN_SLANG_INCLUDE_DIR ${REF_LIN_SLANG_BIN_DIR}/../include)
	set(REF_LIN_SLANG_OUT ${PROJECT_BINARY_DIR}/slang)
	file(MAKE_DIRECTORY ${REF_LIN_SLANG_OUT})
	set(REF_LIN_KERNELS ref_gemm:ref_gemm_f16 ref_trellis:ref_trellis_reconstruct ref_trellis:ref_had128)
	set(REF_LIN_GENERATED "")
	set(REF_LIN_CPU_SOURCES "")
	foreach(item ${REF_LIN_KERNELS})
		string(REPLACE ":" ";" parts ${item})
		list(GET parts 0 sfile)
		list(GET parts 1 entry)
		set(ssrc ${PROJECT_SOURCE_DIR}/src/shaders/${sfile}.slang)
		foreach(tgt cuda:cuda.cu:0 spirv:spv:0 cpp:cpp:1)
			string(REPLACE ":" ";" tparts ${tgt})
			list(GET tparts 0 target)
			list(GET tparts 1 ext)
			list(GET tparts 2 cpu)
			set(sout ${REF_LIN_SLANG_OUT}/${entry}.${ext})
			add_custom_command(OUTPUT ${sout}
				COMMAND ${SLANGC} ${ssrc} -target ${target} -stage compute -entry ${entry} -DENGINE_TARGET_CPU=${cpu} -o ${sout}
				DEPENDS ${ssrc}
				COMMENT "slangc ${entry} -> ${target}")
			list(APPEND REF_LIN_GENERATED ${sout})
		endforeach()
		list(APPEND REF_LIN_CPU_SOURCES ${REF_LIN_SLANG_OUT}/${entry}.cpp)
	endforeach()
	add_custom_target(ref_lin_slang DEPENDS ${REF_LIN_GENERATED})
	add_custom_target(ref_lin_spirv_val DEPENDS ref_lin_slang
		COMMAND spirv-val ${REF_LIN_SLANG_OUT}/ref_gemm_f16.spv
		COMMAND spirv-val ${REF_LIN_SLANG_OUT}/ref_trellis_reconstruct.spv
		COMMAND spirv-val ${REF_LIN_SLANG_OUT}/ref_had128.spv)

	# The main Slang launcher library globs src/shaders/cuda/*.cu: keep the reference launchers out of it
	if(TARGET engine_slang_cuda)
		get_target_property(REF_LIN_SLANG_SRCS engine_slang_cuda SOURCES)
		list(FILTER REF_LIN_SLANG_SRCS EXCLUDE REGEX "/cuda/ref_[^/]*\\.cu$")
		set_property(TARGET engine_slang_cuda PROPERTY SOURCES ${REF_LIN_SLANG_SRCS})
	endif()

	# ref_gemm.cu / ref_trellis.cu include the generated Slang CUDA code; ref_engine.cu bridges to the engine's own Linear
	add_library(ref_lin_cuda STATIC
		src/shaders/cuda/ref_gemm.cu src/shaders/cuda/ref_trellis.cu src/shaders/cuda/ref_engine.cu)
	add_dependencies(ref_lin_cuda ref_lin_slang)
	target_include_directories(ref_lin_cuda PRIVATE ${REF_LIN_SLANG_OUT} ${REF_LIN_SLANG_INCLUDE_DIR} src/shaders/cuda src/kernels/trellis)
	target_link_libraries(ref_lin_cuda PUBLIC engine_kernels)
	set_source_files_properties(src/shaders/cuda/ref_gemm.cu src/shaders/cuda/ref_trellis.cu
		PROPERTIES COMPILE_DEFINITIONS __CUDA_NO_HALF2_OPERATORS__)
	set_target_properties(ref_lin_cuda PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

	set_source_files_properties(${REF_LIN_CPU_SOURCES} PROPERTIES GENERATED TRUE)
	add_executable(test_ref_linear tests/2_parity/test_ref_linear.cpp ${REF_LIN_CPU_SOURCES})
	add_dependencies(test_ref_linear ref_lin_slang)
	target_include_directories(test_ref_linear PRIVATE ${REF_LIN_SLANG_INCLUDE_DIR} src/shaders/cuda)
	target_link_libraries(test_ref_linear PRIVATE ref_lin_cuda)
	set_target_properties(test_ref_linear PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()
# ---- E1 end
