// NVFP4 Linear parity on real Qwen3.8-27B-QUASAR-NVFP4 weights: prepacked skinny kernel vs a dense FP16 dequantization
// of the same slice (load_nvfp4_dense) multiplied in double precision, across row counts and both TP split directions.
#include "engine/runtime/test_paths.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include <cmath>
#include <cstdio>
#include "trellis_context.h"
#include <vector>

static const char* NVFP4_DIR = engine::runtime::test::model_path( "Qwen3.8-27B-QUASAR-NVFP4" );

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 1ULL * 1024 * 1024 * 1024 ), "arena" );
	engine::loader::SafeTensorsRepo repo;
	ENGINE_ASSERT( repo.open( NVFP4_DIR ), "open model dir" );

	struct Case { const char* name; bool split_n; int rank; };
	const Case cases[] = {
		{ "model.language_model.layers.3.mlp.gate_proj", true, 1 },
		{ "model.language_model.layers.3.mlp.down_proj", false, 2 },
		{ "model.language_model.layers.3.self_attn.k_proj", true, 3 },
	};
	bool all_ok = true;
	for ( const Case& c : cases ) {
		pod::LinearWeight lw = repo.load_nvfp4_slice( c.name, arena, c.split_n, c.rank, 4, stream );
		pod::TensorView dense = repo.load_nvfp4_dense( c.name, arena, c.split_n, c.rank, 4, stream );
		ENGINE_ASSERT( lw.is_valid() && dense.data != nullptr, "load failed" );
		const int N = dense.shape[0], K = dense.shape[1];
		std::vector<half> h_w( static_cast<size_t>( N ) * K );
		CUDA_CHECK( cudaMemcpy( h_w.data(), dense.data, h_w.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		for ( int M : { 1, 2, 3, 4, 7, 8, 9, 16, 17, 24, 32, 33, 64, 100, 256, 512 } ) {
			std::vector<half> h_x( static_cast<size_t>( M ) * K );
			uint32_t seed = 777u + M;
			for ( auto& v : h_x ) {
				seed = seed * 1664525u + 1013904223u;
				v = __float2half( ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f );
			}
			half *d_x, *d_y;
			CUDA_CHECK( cudaMalloc( &d_x, h_x.size() * sizeof( half ) ) );
			CUDA_CHECK( cudaMalloc( &d_y, static_cast<size_t>( M ) * N * sizeof( half ) ) );
			CUDA_CHECK( cudaMemcpy( d_x, h_x.data(), h_x.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
			pod::TensorView x{}, y{};
			x.data = d_x; x.dtype = pod::DType::FP16; x.ndim = 2; x.shape[0] = M; x.shape[1] = K;
			y.data = d_y; y.dtype = pod::DType::FP16; y.ndim = 2; y.shape[0] = M; y.shape[1] = N;
			engine::modules::Linear::forward( x, lw, y, stream );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			float us = 0.0f;
			{
				cudaEvent_t e0, e1;
				CUDA_CHECK( cudaEventCreate( &e0 ) );
				CUDA_CHECK( cudaEventCreate( &e1 ) );
				CUDA_CHECK( cudaEventRecord( e0, stream ) );
				for ( int it = 0; it < 20; ++it ) engine::modules::Linear::forward( x, lw, y, stream );
				CUDA_CHECK( cudaEventRecord( e1, stream ) );
				CUDA_CHECK( cudaEventSynchronize( e1 ) );
				CUDA_CHECK( cudaEventElapsedTime( &us, e0, e1 ) );
				us *= 1000.0f / 20.0f;
				(void) cudaEventDestroy( e0 );
				(void) cudaEventDestroy( e1 );
			}
			std::vector<half> h_y( static_cast<size_t>( M ) * N );
			CUDA_CHECK( cudaMemcpy( h_y.data(), d_y, h_y.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			double dot = 0, na = 0, nb = 0, maxabs = 0, maxref = 0;
			const int n_step = M > 16 ? 37 : 1;  // sample columns where the reference gets expensive
			for ( int m = 0; m < M; ++m ) {
				for ( int n = 0; n < N; n += n_step ) {
					double ref = 0;
					for ( int k = 0; k < K; ++k ) ref += static_cast<double>( __half2float( h_x[static_cast<size_t>( m ) * K + k] ) ) * __half2float( h_w[static_cast<size_t>( n ) * K + k] );
					const double got = __half2float( h_y[static_cast<size_t>( m ) * N + n] );
					dot += ref * got; na += ref * ref; nb += got * got;
					maxabs = fmax( maxabs, fabs( ref - got ) );
					maxref = fmax( maxref, fabs( ref ) );
				}
			}
			const double cosine = dot / sqrt( na * nb + 1e-30 );
			const bool ok = cosine >= 0.9999 && maxabs <= 0.005 * maxref + 1e-3;
			all_ok = all_ok && ok;
			std::printf( "  %-52s %s rank %d M=%2d N=%5d K=%5d  cos=%.6f max_abs=%.4f (|ref|max=%.2f) %6.1f us %s\n", c.name + 25, c.split_n ? "N" : "K", c.rank, M, N, K, cosine, maxabs, maxref, us, ok ? "ok" : "FAIL" );
			CUDA_CHECK( cudaFree( d_x ) );
			CUDA_CHECK( cudaFree( d_y ) );
		}
	}
	std::printf( "%s NVFP4 linear parity\n", all_ok ? "[+] PASS" : "[-] FAIL" );
	return all_ok ? 0 : 1;
}
