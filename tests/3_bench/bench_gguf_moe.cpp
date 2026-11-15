// GGUF routed experts (MoEGguf::forward) at decode / verify row counts: time per layer over distinct expert sets (cold L2) in graph replay,
// and the GB/s of the blocks of the experts the routes touch. Routes are uniform over the global experts (fixed seed); this rank holds the
// first `local` of them. The block bytes are a constant pattern, or pseudo-random with BENCH_RANDOM=1 (the IQ grids' lookups then miss as
// real weights do; timing only either way: the values are not meaningful). BENCH_ROWS=a,b,... sets the row counts (prefill: the tile path
// above 64 rows). BENCH_DUMP=<prefix>: pseudo-random inputs, and each row count's output after one forward written to <prefix>_<rows>.bin
// (FP16, rows x hidden), to compare two runs of the same weights (a kernel path A/B; with BENCH_RANDOM=1 for meaningful values). BENCH_REUSE=p: each
// row keeps each of the previous row's experts with probability p (consecutive tokens of a draft share experts), the rest uniform.
//   bench_gguf_moe [TYPE] [hidden inter global local top_k]   default Q2_K 4096 2048 288 72 8 (GLM-5.3-Flash, expert-parallel at TP=4)
//   DeepSeek-V4.1 IQ1_M build: bench_gguf_moe IQ1_S 5120 2304 384 96 6
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/gguf.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <set>
#include <string>
#include <vector>

using engine::modules::GgmlType;

int main( int argc, char** argv ) {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	GgmlType type = GgmlType::Q2_K;
	if ( argc > 1 ) {
		bool found = false;
		for ( uint32_t t = 0; t < 64 && !found; ++t ) {
			if ( engine::modules::gguf_linear_supported( static_cast<GgmlType>( t ) ) && std::strcmp( engine::modules::ggml_type_name( static_cast<GgmlType>( t ) ), argv[1] ) == 0 ) {
				type = static_cast<GgmlType>( t );
				found = true;
			}
		}
		ENGINE_ASSERT( found, "bench_gguf_moe: unknown or unsupported type" );
	}
	const int hidden = argc > 2 ? std::atoi( argv[2] ) : 4096, inter = argc > 3 ? std::atoi( argv[3] ) : 2048;
	const int global = argc > 4 ? std::atoi( argv[4] ) : 288, local = argc > 5 ? std::atoi( argv[5] ) : 72, top_k = argc > 6 ? std::atoi( argv[6] ) : 8;
	const engine::modules::GgmlBlock blk = engine::modules::ggml_block( type );
	const size_t up_bytes = static_cast<size_t>( inter ) * ( hidden / blk.elems ) * blk.bytes, down_bytes = static_cast<size_t>( hidden ) * ( inter / blk.elems ) * blk.bytes;
	const size_t set_bytes = static_cast<size_t>( local ) * ( 2 * up_bytes + down_bytes );
	const int SETS = std::max( 2, std::min( 16, static_cast<int>( 2.5e9 / static_cast<double>( set_bytes ) ) ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, static_cast<size_t>( SETS ) * set_bytes + ( 512ULL << 20 ) ), "arena" );
	// pseudo-random block bytes with FP16 scale fields kept finite (one buffer, copied to every expert)
	std::vector<uint8_t> random_bytes;
	if ( const char* r = std::getenv( "BENCH_RANDOM" ); r != nullptr && std::atoi( r ) != 0 ) {
		random_bytes.resize( std::max( up_bytes, down_bytes ) );
		uint64_t st = 0x9e3779b97f4a7c15ull;
		for ( auto& b : random_bytes ) { st = st * 6364136223846793005ull + 1442695040888963407ull; b = static_cast<uint8_t>( st >> 56 ); }
		int off[2];
		const int fields = engine::modules::ggml_scale_fields( type, off );
		for ( size_t at = 0; at + blk.bytes <= random_bytes.size(); at += blk.bytes ) {
			for ( int f = 0; f < fields; ++f ) { random_bytes[at + off[f]] = 0x00; random_bytes[at + off[f] + 1] = 0x20; }   // 2^-7
			if ( type == GgmlType::IQ1_M ) {   // its FP16 scale is the top nibbles of the four scale words (bytes 48..55): 2^-7 as well
				for ( int w = 0; w < 4; ++w ) random_bytes[at + 49 + 2 * w] = static_cast<uint8_t>( ( random_bytes[at + 49 + 2 * w] & 0x0f ) | ( w == 3 ? 0x20 : 0x00 ) );
			}
		}
	}
	// per set: the gate, up and down pointer arrays
	std::vector<void*> tables( SETS );
	for ( int s = 0; s < SETS; ++s ) {
		std::vector<const void*> ptrs( 3 * local );
		for ( int p = 0; p < 3; ++p ) {
			for ( int e = 0; e < local; ++e ) {
				const size_t b = p < 2 ? up_bytes : down_bytes;
				void* d = engine::core::arena_alloc( arena, b, 256 );
				if ( random_bytes.empty() ) CUDA_CHECK( cudaMemsetAsync( d, 0x11, b, stream ) );
				else CUDA_CHECK( cudaMemcpy( d, random_bytes.data(), b, cudaMemcpyHostToDevice ) );
				ptrs[p * local + e] = d;
			}
		}
		tables[s] = engine::core::arena_alloc( arena, ptrs.size() * sizeof( void* ), 256 );
		CUDA_CHECK( cudaMemcpy( tables[s], ptrs.data(), ptrs.size() * sizeof( void* ), cudaMemcpyHostToDevice ) );
	}
	void* scratch = engine::core::arena_alloc( arena, engine::modules::MoEGguf::scratch_bytes( top_k, hidden, inter ), 256 );
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	std::mt19937 rng( 1 );
	std::printf( "%s hidden %d inter %d, %d of %d experts local, top %d, %d sets\n", engine::modules::ggml_type_name( type ), hidden, inter, local, global, top_k, SETS );
	const double reuse = std::getenv( "BENCH_REUSE" ) ? std::atof( std::getenv( "BENCH_REUSE" ) ) : 0.0;
	std::vector<int> row_counts = { 1, 2, 4, 8, 16 };
	if ( const char* r = std::getenv( "BENCH_ROWS" ) ) {
		row_counts.clear();
		for ( const char* q = r; *q; ) {
			row_counts.push_back( std::atoi( q ) );
			while ( *q && *q != ',' ) ++q;
			if ( *q == ',' ) ++q;
		}
	}
	for ( int m : row_counts ) {
		// distinct experts per row; the local ones the routes touch, for the bytes read
		std::vector<int> topk( static_cast<size_t>( m ) * top_k );
		std::set<int> hit;
		std::set<int> prev;
		for ( int r = 0; r < m; ++r ) {
			std::set<int> row;
			if ( reuse > 0.0 ) for ( int e : prev ) if ( ( rng() % 1000 ) < reuse * 1000 ) row.insert( e );   // the previous row's experts kept
			while ( static_cast<int>( row.size() ) < top_k ) row.insert( static_cast<int>( rng() % global ) );
			prev = row;
			int j = 0;
			for ( int e : row ) {
				topk[static_cast<size_t>( r ) * top_k + j++] = e;
				if ( e < local ) hit.insert( e );
			}
		}
		int* d_topk = static_cast<int*>( engine::core::arena_alloc( arena, topk.size() * sizeof( int ), 256 ) );
		CUDA_CHECK( cudaMemcpy( d_topk, topk.data(), topk.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		std::vector<half> tw( topk.size(), __float2half( 0.125f ) );
		half* d_w = static_cast<half*>( engine::core::arena_alloc( arena, tw.size() * sizeof( half ), 256 ) );
		CUDA_CHECK( cudaMemcpy( d_w, tw.data(), tw.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		half* x = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( m ) * hidden * sizeof( half ), 256 ) );
		half* o = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( m ) * hidden * sizeof( half ), 256 ) );
		const char* dump = std::getenv( "BENCH_DUMP" );
		if ( dump != nullptr ) {
			std::vector<half> xs( static_cast<size_t>( m ) * hidden );
			std::mt19937 xr( 7 + m );
			for ( half& v : xs ) v = __float2half( static_cast<float>( xr() % 2001 ) / 1000.0f - 1.0f );
			CUDA_CHECK( cudaMemcpy( x, xs.data(), xs.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		}
		else CUDA_CHECK( cudaMemsetAsync( x, 0x3c, static_cast<size_t>( m ) * hidden * sizeof( half ), stream ) );
		auto run = [&]() {
			for ( int s = 0; s < SETS; ++s ) {
				engine::modules::MoEGguf::Table t;
				t.ptrs = static_cast<const void* const*>( tables[s] );
				t.num_local = local;
				t.num_global = global;
				t.first_expert = 0;
				t.type[0] = t.type[1] = t.type[2] = type;
				t.hidden = hidden;
				t.inter = inter;
				t.act_limit = 10.0f;
				engine::modules::MoEGguf::forward( t, x, m, d_topk, d_w, top_k, o, scratch, nullptr, stream );
			}
		};
		run();
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		if ( dump != nullptr ) {
			std::vector<half> out( static_cast<size_t>( m ) * hidden );
			CUDA_CHECK( cudaMemcpy( out.data(), o, out.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			const std::string path = std::string( dump ) + "_" + std::to_string( m ) + ".bin";
			FILE* f = std::fopen( path.c_str(), "wb" );
			ENGINE_ASSERT( f != nullptr && std::fwrite( out.data(), sizeof( half ), out.size(), f ) == out.size(), "bench_gguf_moe: BENCH_DUMP write" );
			std::fclose( f );
		}
		cudaGraph_t g;
		cudaGraphExec_t ex;
		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		run();
		CUDA_CHECK( cudaStreamEndCapture( stream, &g ) );
		CUDA_CHECK( cudaGraphInstantiate( &ex, g, nullptr, nullptr, 0 ) );
		CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		const int iters = 20;
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		for ( int it = 0; it < iters; ++it ) CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		const double us = 1000.0 * ms / iters / SETS, read = static_cast<double>( hit.size() ) * ( 2 * up_bytes + down_bytes );
		std::printf( "  rows %4d: %3zu local experts hit, %9.2f us per layer, %6.1f GB/s of their blocks\n", m, hit.size(), us, read / us / 1e3 );
		CUDA_CHECK( cudaGraphExecDestroy( ex ) );
		CUDA_CHECK( cudaGraphDestroy( g ) );
	}
	return 0;
}
