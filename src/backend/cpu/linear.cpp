// CPU backend: dense FP16 GEMV/GEMM (the Linear FP16 path). Trellis is in linear_trellis.cpp; NVFP4 is in linear_nvfp4.cpp.
#include "cpu_kernels.h"
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
	impl::cpu_parallel_for( n, 64, [&]( int64_t c0, int64_t c1 ) {
		for ( int64_t c = c0; c < c1; ++c ) {
			const backend::half_t* wr = w + static_cast<size_t>( c ) * k;
			for ( int r = 0; r < m; ++r ) {
				const backend::half_t* xr = x + static_cast<size_t>( r ) * k;
				float a = 0.0f;
				for ( int i = 0; i < k; ++i ) a += static_cast<float>( xr[i] ) * static_cast<float>( wr[i] );
				out[static_cast<size_t>( r ) * n + c] = static_cast<backend::half_t>( a );
			}
		}
	} );
}

} // namespace engine::modules
