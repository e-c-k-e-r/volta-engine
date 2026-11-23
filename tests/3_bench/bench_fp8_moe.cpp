// FP8 routed experts (FP8MoE::forward) at prefill row counts: time per layer and the TFLOP/s of the routed products, routes uniform over
// the global experts (fixed seed), this rank holding the first `local`. Weights are pseudo-random E4M3 bytes (no NaN codes) with unit scales;
// timing only. BENCH_ROWS=a,b,... sets the row counts.
//   bench_fp8_moe [hidden inter global local top_k]   default 4096 2048 288 72 8 (GLM-5.3-Flash, expert-parallel at TP=4)
#include "engine/config.h"
#include "engine/modules/fp8_moe.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

int main( int argc, char** argv ) {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	const int hidden = argc > 1 ? std::atoi( argv[1] ) : 4096, inter = argc > 2 ? std::atoi( argv[2] ) : 2048;
	const int global = argc > 3 ? std::atoi( argv[3] ) : 288, local = argc > 4 ? std::atoi( argv[4] ) : 72, top_k = argc > 5 ? std::atoi( argv[5] ) : 8;
	const size_t proj = static_cast<size_t>( hidden ) * inter;
	std::vector<uint8_t> bytes( proj );
	uint64_t st = 0x9e3779b97f4a7c15ull;
	for ( auto& b : bytes ) { do { st = st * 6364136223846793005ull + 1442695040888963407ull; b = static_cast<uint8_t>( st >> 56 ); } while ( ( b & 0x7f ) == 0x7f ); }
	std::vector<float> ones( proj / ( 128 * 128 ), 1.0f / 64 );
	std::vector<const void*> table( 6 * static_cast<size_t>( local ) );
	for ( int e = 0; e < local; ++e ) {
		for ( int p = 0; p < 3; ++p ) {
			void* c = nullptr, * s = nullptr;
			CUDA_CHECK( cudaMalloc( &c, proj ) );
			CUDA_CHECK( cudaMemcpy( c, bytes.data(), proj, cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMalloc( &s, ones.size() * sizeof( float ) ) );
			CUDA_CHECK( cudaMemcpy( s, ones.data(), ones.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			table[( 2 * p ) * local + e] = c;
			table[( 2 * p + 1 ) * local + e] = s;
		}
	}
	void* d_table = nullptr;
	CUDA_CHECK( cudaMalloc( &d_table, table.size() * sizeof( void* ) ) );
	CUDA_CHECK( cudaMemcpy( d_table, table.data(), table.size() * sizeof( void* ), cudaMemcpyHostToDevice ) );
	const void* const* tp = static_cast<const void* const*>( d_table );
	engine::modules::FP8ExpertTable t;
	t.gate_c = reinterpret_cast<const uint8_t* const*>( tp + 0 * local ); t.gate_s = reinterpret_cast<const float* const*>( tp + 1 * local );
	t.up_c = reinterpret_cast<const uint8_t* const*>( tp + 2 * local );   t.up_s = reinterpret_cast<const float* const*>( tp + 3 * local );
	t.down_c = reinterpret_cast<const uint8_t* const*>( tp + 4 * local ); t.down_s = reinterpret_cast<const float* const*>( tp + 5 * local );
	t.num_local = local;
	t.min_expert = 0;
	t.num_global = global;
	t.hidden = hidden;
	t.inter = inter;
	void* scratch = nullptr;
	CUDA_CHECK( cudaMalloc( &scratch, engine::modules::FP8MoE::scratch_bytes( top_k, hidden, inter ) ) );
	std::vector<int> row_counts = { 256, 1024, 4096 };
	if ( const char* r = std::getenv( "BENCH_ROWS" ) ) {
		row_counts.clear();
		for ( const char* q = r; *q; ) {
			row_counts.push_back( std::atoi( q ) );
			while ( *q && *q != ',' ) ++q;
			if ( *q == ',' ) ++q;
		}
	}
	std::mt19937 rng( 1 );
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	std::printf( "FP8 hidden %d inter %d, %d of %d experts local, top %d\n", hidden, inter, local, global, top_k );
	for ( int m : row_counts ) {
		std::vector<int> topk( static_cast<size_t>( m ) * top_k );
		size_t routes_local = 0;
		for ( int r = 0; r < m; ++r ) {
			std::set<int> row;
			while ( static_cast<int>( row.size() ) < top_k ) row.insert( static_cast<int>( rng() % global ) );
			int j = 0;
			for ( int e : row ) { topk[static_cast<size_t>( r ) * top_k + j++] = e; routes_local += e < local; }
		}
		int* d_topk = nullptr;
		half* d_w = nullptr, * x = nullptr, * o = nullptr;
		CUDA_CHECK( cudaMalloc( &d_topk, topk.size() * sizeof( int ) ) );
		CUDA_CHECK( cudaMemcpy( d_topk, topk.data(), topk.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		std::vector<half> tw( topk.size(), __float2half( 0.125f ) );
		CUDA_CHECK( cudaMalloc( &d_w, tw.size() * sizeof( half ) ) );
		CUDA_CHECK( cudaMemcpy( d_w, tw.data(), tw.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMalloc( &x, static_cast<size_t>( m ) * hidden * sizeof( half ) ) );
		CUDA_CHECK( cudaMalloc( &o, static_cast<size_t>( m ) * hidden * sizeof( half ) ) );
		CUDA_CHECK( cudaMemset( x, 0x2c, static_cast<size_t>( m ) * hidden * sizeof( half ) ) );
		engine::modules::FP8MoE::forward( t, x, m, d_topk, d_w, top_k, o, scratch, nullptr, 10.0f, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		const int iters = 10;
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		for ( int it = 0; it < iters; ++it ) engine::modules::FP8MoE::forward( t, x, m, d_topk, d_w, top_k, o, scratch, nullptr, 10.0f, stream );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		const double per = ms / iters, flop = 2.0 * 3.0 * static_cast<double>( proj ) * routes_local;
		std::printf( "  rows %5d: %6zu local routes, %8.3f ms per layer, %6.1f TFLOP/s\n", m, routes_local, per, flop / per / 1e9 );
		CUDA_CHECK( cudaFree( d_topk ) );
		CUDA_CHECK( cudaFree( d_w ) );
		CUDA_CHECK( cudaFree( x ) );
		CUDA_CHECK( cudaFree( o ) );
	}
	return 0;
}
