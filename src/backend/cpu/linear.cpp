// CPU backend: dense FP16 GEMV/GEMM (the Linear FP16 path). Trellis is in linear_trellis.cpp; NVFP4 stays CUDA-only.
#include "engine/modules/linear.h"

namespace engine::modules {

// out[m][n] = sum_k x[m][k] * w[n][k], FP32 accumulation; w is [n, k] row-major in memory (the loader's layout, tensor shape
// tagged [k, n]), the same as dflash_f16_gemv_nt on CUDA
void LinearPlainGEMV::forward(
	const backend::half_t* w,
	const backend::half_t* x,
	backend::half_t* out,
	int m,
	int n,
	int k,
	backend::stream_t
) {
	float* acc = new float[static_cast<size_t>( n )];
	for ( int r = 0; r < m; ++r ) {
		const backend::half_t* xr = x + static_cast<size_t>( r ) * k;
		for ( int c = 0; c < n; ++c ) {
			const backend::half_t* wr = w + static_cast<size_t>( c ) * k;
			float a = 0.0f;
			for ( int i = 0; i < k; ++i ) a += static_cast<float>( xr[i] ) * static_cast<float>( wr[i] );
			acc[c] = a;
		}
		for ( int c = 0; c < n; ++c ) out[static_cast<size_t>( r ) * n + c] = static_cast<backend::half_t>( acc[c] );
	}
	delete[] acc;
}

} // namespace engine::modules
