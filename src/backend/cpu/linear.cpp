// CPU backend: dense FP16 GEMV/GEMM (the Linear FP16 path). Quantized formats (Trellis, NVFP4) stay CUDA-only.
#include "engine/modules/linear.h"

namespace engine::modules {

// out[m][n] = sum_k x[m][k] * w[k][n], FP32 accumulation; w is [k, n] row-major
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
		for ( int c = 0; c < n; ++c ) acc[c] = 0.0f;
		const backend::half_t* xr = x + static_cast<size_t>( r ) * k;
		for ( int i = 0; i < k; ++i ) {
			const float xv = static_cast<float>( xr[i] );
			const backend::half_t* wr = w + static_cast<size_t>( i ) * n;
			for ( int c = 0; c < n; ++c ) acc[c] += xv * static_cast<float>( wr[c] );
		}
		for ( int c = 0; c < n; ++c ) out[static_cast<size_t>( r ) * n + c] = static_cast<backend::half_t>( acc[c] );
	}
	delete[] acc;
}

} // namespace engine::modules
