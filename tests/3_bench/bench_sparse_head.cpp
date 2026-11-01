// What a sparse drafter head would save on one rank: the Flash-Next head's real tensor-parallel shard (EXL3, 6 bits, its padded columns) through
// Linear::forward for 1 to 4 draft rows, against a plain FP16 GEMV over a subset of the vocabulary's columns (this rank's quarter of C columns)
// for C = 4096 .. 32768. Timing only. Run on one GPU: CUDA_VISIBLE_DEVICES=1 bench_sparse_head [model dir]
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/runtime/test_paths.h"
#include "engine/spec/dflash.h"
#include "exl3_gemm.cuh"
#include "trellis_context.h"
#include <cstdio>
#include <string>
#include <vector>

template <typename F>
static float time_ms( F&& f, cudaStream_t s, int iters ) {
	for ( int i = 0; i < 5; ++i ) f();
	cudaEvent_t a, b;
	CUDA_CHECK( cudaEventCreate( &a ) );
	CUDA_CHECK( cudaEventCreate( &b ) );
	CUDA_CHECK( cudaEventRecord( a, s ) );
	for ( int i = 0; i < iters; ++i ) f();
	CUDA_CHECK( cudaEventRecord( b, s ) );
	CUDA_CHECK( cudaEventSynchronize( b ) );
	float ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &ms, a, b ) );
	return ms / iters;
}

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const char* dir = argc > 1 ? argv[1] : engine::runtime::test::model_path( "ENGINE_TEST_FLASH_DIR" );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	exl3_gemm_init_all_attrs( 0 );
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 3ULL << 30 ), "arena" );
	engine::loader::SafeTensorsRepo repo;
	ENGINE_ASSERT( repo.open( dir ), "cannot open the model directory" );
	const int tp = 4, hidden = 2560, vocab = 248320;
	const int real = vocab / tp, cols = real + ( 256 - real % 256 ) % 256;   // as Qwen4ExpPipeline pads its shard
	const pod::LinearWeight head = repo.load_linear_slice_padded( "lm_head", arena, 6, 0, tp, cols - real, s );
	ENGINE_ASSERT( head.is_valid(), "lm_head shard not loaded" );
	CUDA_CHECK( cudaStreamSynchronize( s ) );
	const int shape_x[2] = { 4, hidden }, shape_o[2] = { 4, cols };
	pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape_x, 2 );
	pod::TensorView o = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, shape_o, 2 );
	CUDA_CHECK( cudaMemsetAsync( x.data, 0, 4 * hidden * sizeof( half ), s ) );
	const double head_mb = static_cast<double>( hidden ) * cols * 6 / 8 / 1e6;
	std::printf( "head shard %d x %d at 6 bits (%.1f MB)\n", hidden, cols, head_mb );
	for ( int m : { 1, 2, 4 } ) {
		pod::TensorView xm = x, om = o;
		xm.shape[0] = om.shape[0] = m;
		const float t = time_ms( [&]() { engine::modules::Linear::forward( xm, head, om, s ); }, s, 200 );
		std::printf( "  EXL3 head shard, %d row(s): %7.1f us (%5.0f GB/s)\n", m, t * 1e3, head_mb / t );
	}
	half* w;
	const int max_c = 32768 / tp;
	CUDA_CHECK( cudaMalloc( &w, static_cast<size_t>( max_c ) * hidden * sizeof( half ) ) );
	CUDA_CHECK( cudaMemsetAsync( w, 0, static_cast<size_t>( max_c ) * hidden * sizeof( half ), s ) );
	for ( int c : { 4096, 8192, 16384, 32768 } ) {
		const int n = c / tp;
		for ( int m : { 1, 4 } ) {
			const float t = time_ms( [&]() { engine::spec::dflash_f16_gemv_nt( w, static_cast<const half*>( x.data ), static_cast<half*>( o.data ), m, n, hidden, s ); }, s, 200 );
			std::printf( "  FP16 subset of %5d columns (%4d per rank, %.1f MB), %d row(s): %7.1f us\n", c, n, static_cast<double>( n ) * hidden * 2 / 1e6, m, t * 1e3 );
		}
	}
	return 0;
}
