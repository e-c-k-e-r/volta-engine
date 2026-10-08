#pragma once

// CUDA backend vocabulary (selected by engine/backend/types.h): aliases of the CUDA runtime types
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace engine::backend {
	using stream_t = cudaStream_t;
	using graph_t = cudaGraph_t;
	using graph_exec_t = cudaGraphExec_t;
	using half_t = __half;
	using bf16_t = __nv_bfloat16;

	// Direction of a copy; the call sites name it instead of the backend's enum
	// Auto lets the runtime infer the direction from the pointers (unified addressing), as for loads from mapped files
	enum class Copy { H2D, D2H, D2D, Auto };

	// Pinned host allocation flavors
	enum class Pin { Default, Mapped, MappedPortable };
}
