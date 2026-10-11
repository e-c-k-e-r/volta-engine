// Round-to-nearest NVFP4 (SafeTensorsRepo::quantize_nvfp4_slice from a BF16 tensor) against another representation of the same matrix, same inputs:
// the EXL3 trellis tensor of an EXL3 checkpoint. Reports the cosine and the relative difference of the outputs at 1 and 8 rows.
//   test_nvfp4_quantize <exl3_dir> <nvfp4_dir> <tensor base name> [bits]
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "trellis_context.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 4 ) { std::printf( "usage: test_nvfp4_quantize <exl3_dir> <nvfp4_dir> <tensor base name> [bits]\n" ); return 1; }
	const int bits = argc > 4 ? std::atoi( argv[4] ) : 4;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	engine::modules::trellis_get_workspace( 0, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( 0 );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 1ULL << 30 ), "arena" );
	engine::loader::SafeTensorsRepo ex, nv;
	ENGINE_ASSERT( ex.open( argv[1] ), "exl3 dir" );
	ENGINE_ASSERT( nv.open( argv[2] ), "nvfp4 dir" );
	pod::LinearWeight tr = ex.load_linear( argv[3], arena, bits, nullptr );
	pod::LinearWeight fp = nv.quantize_nvfp4_slice( argv[3], arena, true, 0, 1, 0, nullptr );
	ENGINE_ASSERT( tr.is_valid() && fp.is_valid(), "tensors" );
	// K from the trellis shape [K/16, N/16, 16 * bits], N likewise
	const int K = tr.w0.shape[0] * 16, N = tr.w0.shape[1] * 16;
	for ( int M : { 1, 8 } ) {
		std::vector<half> hx( static_cast<size_t>( M ) * K );
		uint32_t seed = 99u;
		for ( auto& v : hx ) { seed = seed * 1664525u + 1013904223u; v = __float2half( ( ( seed >> 8 ) / 16777216.0f - 0.5f ) * 4.0f ); }
		int xs[2] = { M, K }, os[2] = { M, N };
		pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		pod::TensorView o1 = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
		pod::TensorView o2 = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
		CUDA_CHECK( cudaMemcpy( x.data, hx.data(), hx.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		engine::modules::Linear::forward( x, tr, o1 );
		engine::modules::Linear::forward( x, fp, o2 );
		CUDA_CHECK( cudaDeviceSynchronize() );
		std::vector<half> a( static_cast<size_t>( M ) * N ), b( a.size() );
		CUDA_CHECK( cudaMemcpy( a.data(), o1.data, a.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( b.data(), o2.data, b.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		double dot = 0, na = 0, nb = 0, nd = 0;
		for ( size_t i = 0; i < a.size(); ++i ) {
			const double p = __half2float( a[i] ), q = __half2float( b[i] );
			dot += p * q; na += p * p; nb += q * q; nd += ( p - q ) * ( p - q );
		}
		std::printf( "%s K %d N %d M=%d: cosine %.6f, |exl3| %.4g, |nvfp4| %.4g, relative difference %.4g\n", argv[3], K, N, M, dot / std::sqrt( na * nb ), std::sqrt( na ), std::sqrt( nb ), std::sqrt( nd / na ) );
	}
	return 0;
}
