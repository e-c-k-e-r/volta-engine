// GLM-5.3-Flash linear-attention qkv_proj, the way the engine loads it: one rank's slice of the fused trellis tensor (three groups of 512 tile columns, one slice of
// each) through the engine's loader and Linear::forward, against the rows of the source's q_proj / k_proj / v_proj that rank owns (stacked q, k, v).
//   test_glm_qkv <exl3 checkpoint dir> <source dir> [layer] [rank]
// pass: the relative error is that of a 5 bit quantization (a wrong group order or slice gives about 1.4).
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main( int argc, char** argv ) {
	if ( argc < 3 ) { std::printf( "usage: test_glm_qkv <exl3 dir> <source dir> [layer] [rank]\n" ); return 2; }
	const int layer = argc > 3 ? std::atoi( argv[3] ) : 4, rank = argc > 4 ? std::atoi( argv[4] ) : 1, tp = 4;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	engine::modules::trellis_get_workspace( 0, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( 0 );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 3ULL * 1024 * 1024 * 1024 ), "arena" );
	engine::loader::SafeTensorsRepo repo, src;
	ENGINE_ASSERT( repo.open( argv[1] ), "open exl3 dir" );
	ENGINE_ASSERT( src.open( argv[2] ), "open source dir" );

	const int blocks[3] = { 512, 512, 512 }, rank_blocks[3] = { 512 / tp, 512 / tp, 512 / tp };
	char name[300];
	std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.self_attn.qkv_proj", layer );
	pod::LinearWeight w = repo.load_grouped_trellis_slice( name, arena, blocks, rank_blocks, 3, rank );
	ENGINE_ASSERT( w.is_valid(), "qkv slice did not load" );

	const int K = 4096, per = 8192 / tp, N = 3 * per, M = 4;
	std::vector<float> hx( static_cast<size_t>( M ) * K );
	uint32_t st = 12345u;
	for ( auto& v : hx ) { st = st * 1664525u + 1013904223u; v = ( static_cast<int>( st >> 16 ) % 2001 - 1000 ) / 500.0f; }
	std::vector<half> hxh( hx.size() );
	for ( size_t i = 0; i < hx.size(); ++i ) hxh[i] = __float2half( hx[i] );
	half* dx = nullptr; half* dy = nullptr;
	CUDA_CHECK( cudaMalloc( &dx, hxh.size() * sizeof( half ) ) );
	CUDA_CHECK( cudaMalloc( &dy, static_cast<size_t>( M ) * N * sizeof( half ) ) );
	CUDA_CHECK( cudaMemcpy( dx, hxh.data(), hxh.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
	pod::TensorView x = pod::make_tensor_2d( dx, pod::DType::FP16, M, K ), y = pod::make_tensor_2d( dy, pod::DType::FP16, M, N );
	engine::modules::Linear::forward( x, w, y, stream );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	std::vector<half> hy( static_cast<size_t>( M ) * N );
	CUDA_CHECK( cudaMemcpy( hy.data(), dy, hy.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );

	// reference: this rank's rows of q, k, v
	std::vector<float> ref( static_cast<size_t>( M ) * N, 0.0f );
	const char* parts[3] = { "q_proj", "k_proj", "v_proj" };
	for ( int g = 0; g < 3; ++g ) {
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.self_attn.%s.weight", layer, parts[g] );
		pod::TensorView t = src.load( name, arena, stream, pod::DType::FP16 );
		ENGINE_ASSERT( t.data != nullptr, "source tensor missing" );
		std::vector<half> hw( static_cast<size_t>( per ) * K );
		CUDA_CHECK( cudaMemcpy( hw.data(), static_cast<const half*>( t.data ) + static_cast<size_t>( rank ) * per * K, hw.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		for ( int r = 0; r < per; ++r )
			for ( int m = 0; m < M; ++m ) {
				double acc = 0.0;
				for ( int k = 0; k < K; ++k ) acc += static_cast<double>( __half2float( hw[static_cast<size_t>( r ) * K + k] ) ) * hx[static_cast<size_t>( m ) * K + k];
				ref[static_cast<size_t>( m ) * N + g * per + r] = static_cast<float>( acc );
			}
	}
	double num = 0.0, den = 0.0, dot = 0.0, yy = 0.0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const double a = __half2float( hy[i] ), b = ref[i];
		num += ( a - b ) * ( a - b ); den += b * b; dot += a * b; yy += a * a;
	}
	const double rel = std::sqrt( num / den ), cs = dot / std::sqrt( yy * den );
	std::printf( "[*] layer %d rank %d qkv_proj slice: relative error %.4f cosine %.5f  (a 5 bit quantization gives about 0.04)\n", layer, rank, rel, cs );
	const bool ok = rel < 0.1;
	std::printf( ok ? "[+] test_glm_qkv PASS\n" : "[-] test_glm_qkv FAIL\n" );
	return ok ? 0 : 1;
}
