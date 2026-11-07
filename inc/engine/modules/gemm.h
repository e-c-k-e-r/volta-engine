#pragma once

#include "engine/backend/types.h"

// Dense FP16 GEMM on the tensor cores with FP32 accumulation, the engine's own (no cuBLAS): the prefill-size linears and products the decode
// kernels do not cover. Row-major operands.
namespace engine::modules {

struct Gemm {
	// out[m, n] = beta * out + a[m, k] * w[n, k]^T (a linear: w stores one output's weights per row). lda / ldw / ldc in elements; out FP16 or
	// FP32 (out_f32). k a multiple of 8
	static void tn( const backend::half_t* a, int lda, const backend::half_t* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k,
		float beta = 0.0f, backend::stream_t stream = nullptr );
	// out[m, n] = beta * out + a[m, k] * w[k, n] (weights stored k-major, one input's row of outputs per row). n a multiple of 8 as well
	static void nn( const backend::half_t* a, int lda, const backend::half_t* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k,
		float beta = 0.0f, backend::stream_t stream = nullptr );
	// Forgets the split-k workspaces and launch memos after a device reset (trellis_reset_launch_caches)
	static void reset_caches();
};

} // namespace engine::modules
