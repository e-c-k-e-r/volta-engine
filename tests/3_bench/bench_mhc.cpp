// GLM decode hyper-connection: time of HyperConnection::collapse_norm (mix, sinkhorn, collapse, RMSNorm) for 1..8 rows over 24 distinct
// fn copies (cycled so the L2 does not hold them), the way a decode step meets its 156 sites; prints a checksum of the outputs.
//   bench_mhc [rows ...]
#include "engine/core/arena.h"
#include "engine/modules/mhc.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const int D = 4096, HC = 4, SITES = 24;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 1ULL * 1024 * 1024 * 1024 ), "arena" );
	std::vector<int> rows_list;
	for ( int i = 1; i < argc; ++i ) rows_list.push_back( std::atoi( argv[i] ) );
	if ( rows_list.empty() ) rows_list = { 1, 4, 8 };
	uint32_t seed = 99u;
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; };
	auto upload = [&]( pod::DType t, int n, float scale, float offset, bool half_t ) {
		int sh[1] = { n };
		auto v = engine::core::arena_alloc_tensor( arena, t, sh, 1 );
		if ( half_t ) {
			std::vector<half> h( n );
			for ( auto& e : h ) e = __float2half( offset + rnd() * scale );
			CUDA_CHECK( cudaMemcpy( v.data, h.data(), n * sizeof( half ), cudaMemcpyHostToDevice ) );
		}
		else {
			std::vector<float> h( n );
			for ( auto& e : h ) e = offset + rnd() * scale;
			CUDA_CHECK( cudaMemcpy( v.data, h.data(), n * sizeof( float ), cudaMemcpyHostToDevice ) );
		}
		return v;
	};
	std::vector<pod::TensorView> fn( SITES ), base( SITES ), scale( SITES );
	for ( int s = 0; s < SITES; ++s ) {
		fn[s] = upload( pod::DType::FP32, 24 * HC * D, 0.02f, 0.0f, false );
		base[s] = upload( pod::DType::FP32, 24, 0.5f, 0.0f, false );
		scale[s] = upload( pod::DType::FP32, 3, 0.1f, 0.5f, false );
	}
	auto norm_w = upload( pod::DType::FP16, D, 0.1f, 1.0f, true );
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	for ( int rows : rows_list ) {
		auto streams = upload( pod::DType::FP32, rows * HC * D, 1.0f, 0.0f, false );
		auto norm_out = upload( pod::DType::FP16, rows * D, 0.0f, 0.0f, true );
		auto post = upload( pod::DType::FP32, rows * HC, 0.0f, 0.0f, false );
		auto comb = upload( pod::DType::FP32, rows * HC * HC, 0.0f, 0.0f, false );
		auto inv = upload( pod::DType::FP32, rows, 0.0f, 0.0f, false );
		auto logits = upload( pod::DType::FP32, rows * 24, 0.0f, 0.0f, false );
		auto pre = upload( pod::DType::FP32, rows * HC, 0.0f, 0.0f, false );
		auto run = [&]() {
			for ( int s = 0; s < SITES; ++s )
				ENGINE_ASSERT( engine::modules::HyperConnection::collapse_norm( streams, fn[s], base[s], scale[s], norm_w, norm_out, post, comb, inv, logits, pre, D, std::getenv( "SINK" ) ? std::atoi( std::getenv( "SINK" ) ) : 20, 1e-6f, 1e-6f, 0.0f, stream, rows ), "collapse_norm" );
		};
		for ( int i = 0; i < 5; ++i ) run();
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		const int iters = 100;
		cudaGraph_t graph;
		cudaGraphExec_t exec;
		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		run();
		CUDA_CHECK( cudaStreamEndCapture( stream, &graph ) );
		CUDA_CHECK( cudaGraphInstantiate( &exec, graph, nullptr, nullptr, 0 ) );
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		for ( int i = 0; i < iters; ++i ) CUDA_CHECK( cudaGraphLaunch( exec, stream ) );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		std::vector<half> ho( rows * D );
		std::vector<float> hp( rows * HC ), hc( rows * HC * HC );
		CUDA_CHECK( cudaMemcpy( ho.data(), norm_out.data, ho.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( hp.data(), post.data, hp.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
		CUDA_CHECK( cudaMemcpy( hc.data(), comb.data, hc.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
		double sum = 0.0, sp = 0.0, sc = 0.0;
		for ( auto v : ho ) sum += __half2float( v );
		for ( auto v : hp ) sp += v;
		for ( auto v : hc ) sc += v;
		std::printf( "rows %d: %.2f us per site, graph replay; checksum norm %.6f post %.6f comb %.6f\n", rows, ms * 1000.0f / iters / SITES, sum, sp, sc );
	}
	return 0;
}
