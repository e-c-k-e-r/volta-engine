// GGUF linear (LinearGguf, CUDA and Vulkan builds) at decode / verify shapes: time per call over distinct weight sets (cold L2) in graph replay, and the GB/s of
// the blocks read. The block bytes are a constant pattern (timing only: the values are not meaningful).
//   bench_gguf_shapes [TYPE ...] [-- K N ...]   default types Q4_K Q6_K IQ3_S IQ4_XS, default shapes the 27B's per-rank projections at TP=4
//   BENCH_DEQUANT=1 also times LinearGguf::dequant of the same weights (llama.cpp's converter: the decode alone, to FP16 in global memory)
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/gguf.h"
#include "engine/backend/runtime.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

using engine::modules::GgmlType;
namespace backend = engine::backend;

int main( int argc, char** argv ) {
	backend::set_device( 0 );
	backend::stream_t stream = backend::stream_create();
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 3ULL * 1024 * 1024 * 1024 ), "arena" );
	std::vector<GgmlType> types;
	std::vector<std::pair<int, int>> shapes;
	int i = 1;
	for ( ; i < argc && std::strcmp( argv[i], "--" ) != 0; ++i ) {
		bool found = false;
		for ( uint32_t t = 0; t < 64 && !found; ++t ) {
			if ( engine::modules::gguf_linear_supported( static_cast<GgmlType>( t ) ) && std::strcmp( engine::modules::ggml_type_name( static_cast<GgmlType>( t ) ), argv[i] ) == 0 ) {
				types.push_back( static_cast<GgmlType>( t ) );
				found = true;
			}
		}
		ENGINE_ASSERT( found, "bench_gguf_shapes: unknown or unsupported type" );
	}
	for ( ++i; i + 1 < argc; i += 2 ) shapes.push_back( { std::atoi( argv[i] ), std::atoi( argv[i + 1] ) } );
	if ( types.empty() ) types = { GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::IQ3_S, GgmlType::IQ4_XS };
	// in_proj_qkv, gate / up, down, out_proj of one rank at TP=4
	if ( shapes.empty() ) shapes = { { 5120, 2560 }, { 5120, 4352 }, { 4352, 5120 }, { 1536, 5120 } };
	engine::modules::LinearGguf::prepare();
	auto timed = [&]( auto&& body ) {   // ms of body, the stream drained before and after
		backend::stream_sync( stream );
		const auto t0 = std::chrono::steady_clock::now();
		body();
		backend::stream_sync( stream );
		return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
	};
	for ( GgmlType type : types ) {
		const engine::modules::GgmlBlock blk = engine::modules::ggml_block( type );
		for ( auto [K, N] : shapes ) {
			if ( K % blk.elems != 0 ) continue;
			const size_t bytes = static_cast<size_t>( N ) * ( K / blk.elems ) * blk.bytes;
			const int SETS = std::max( 2, std::min( 64, static_cast<int>( 2.0e9 / static_cast<double>( bytes ) ) ) );
			std::vector<pod::TensorView> w( SETS );
			int ws[1] = { static_cast<int>( bytes ) };
			for ( int s = 0; s < SETS; ++s ) {
				w[s] = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, ws, 1 );
				backend::memset_async( w[s].data, 0x11, bytes, stream );
			}
			for ( int m : { 1, 2, 4, 8, 16 } ) {
				int xs[2] = { m, K }, os[2] = { m, N };
				pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
				pod::TensorView o = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
				backend::memset_async( x.data, 0x3c, static_cast<size_t>( m ) * K * 2, stream );
				auto run = [&]() { for ( int s = 0; s < SETS; ++s ) engine::modules::LinearGguf::forward( x, w[s].data, type, N, K, o, stream ); };
				run();
				backend::stream_sync( stream );
				backend::graph_capture_begin( stream );
				run();
				backend::graph_t g = backend::graph_capture_end( stream );
				backend::graph_exec_t ex = backend::graph_instantiate( g );
				backend::graph_launch( ex, stream );
				const int iters = 20;
				const double ms = timed( [&] { for ( int it = 0; it < iters; ++it ) backend::graph_launch( ex, stream ); } );
				const double us = 1000.0 * ms / iters / SETS;
				std::printf( "%-7s K %5d N %5d M %2d: %7.2f us per call, %6.1f GB/s (blocks %.2f MB)\n", engine::modules::ggml_type_name( type ), K, N, m, us,
					static_cast<double>( bytes ) / ( us * 1e-6 ) / 1e9, static_cast<double>( bytes ) / 1e6 );
				backend::graph_exec_destroy( ex );
				backend::graph_destroy( g );
			}
			if ( std::getenv( "BENCH_DEQUANT" ) ) {
				int ds[1] = { N * K };
				pod::TensorView d = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ds, 1 );
				const int reps = std::min( SETS, 8 );
				auto run = [&]() { for ( int s = 0; s < reps; ++s ) engine::modules::LinearGguf::dequant( w[s].data, type, static_cast<int64_t>( N ) * K, d.ptr<backend::half_t>(), stream ); };
				run();
				const double ms = timed( [&] { for ( int it = 0; it < 10; ++it ) run(); } );
				std::printf( "%-7s K %5d N %5d dequant: %7.2f us per call\n", engine::modules::ggml_type_name( type ), K, N, 1000.0 * ms / 10 / reps );
			}
			engine::core::arena_reset( arena );
		}
	}
	return 0;
}
