// The TTS operators at the v2.0 export's shapes (engine/modules/tts.h, CUDA and Vulkan builds): the backbone's GEMMs over a pass of L positions
// (FP16 or FP32 out), the attention of one row, and the codec's convolutions; time per call and TFLOP/s. The operands are constant patterns
// (timing only).
//   bench_tts_ops [L] [reps]   default L 946 (a 4.5 s line's three guidance rows), 10 reps
#include "engine/backend/runtime.h"
#include "engine/core/arena.h"
#include "engine/modules/tts.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

namespace backend = engine::backend;
using engine::modules::CodecOps;
using engine::modules::TtsOps;

static double time_ms( const std::function<void()>& f, int reps ) {
	f();
	backend::device_sync();
	const auto t0 = std::chrono::steady_clock::now();
	for ( int r = 0; r < reps; ++r ) f();
	backend::device_sync();
	return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() / reps;
}

int main( int argc, char** argv ) {
	const int L = argc > 1 ? std::atoi( argv[1] ) : 946;
	const int reps = argc > 2 ? std::atoi( argv[2] ) : 10;
	backend::set_device( 0 );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 2ULL << 30 ), "arena" );
	auto A = [&]( size_t bytes ) {
		void* p = engine::core::arena_alloc( arena, bytes );
		backend::memset_sync( p, 0x11, bytes / 4 * 4 );   // small FP16 / FP32 values, no NaNs
		return p;
	};
	const int d = 1536, F = 4096, H = 24, hd = 64;
	auto* x = static_cast<backend::half_t*>( A( static_cast<size_t>( L ) * F * 2 ) );
	const size_t w_elems = static_cast<size_t>( 2 * F ) * d * 3;   // the extra shapes' room too
	auto* w = static_cast<backend::half_t*>( A( w_elems * 2 ) );
	void* c = A( static_cast<size_t>( L ) * 3 * F * 4 );
	struct G { const char* name; int N, K; bool c32; };
	const G gemms[] = { { "qkv", 3 * d, d, false }, { "gate", d, d, false }, { "out", d, d, true }, { "up", 2 * F, d, false }, { "down", d, F, true } };
	std::printf( "GEMMs over %d positions:\n", L );
	for ( const G& g : gemms ) {
		const double ms = time_ms( [&] { TtsOps::gemm( x, w, c, L, g.N, g.K, g.c32 ); }, reps );
		std::printf( "  %-5s %5d x %5d x %5d %s: %8.3f ms  %6.2f TFLOP/s\n", g.name, L, g.N, g.K, g.c32 ? "f32" : "f16", ms, 2.0 * L * g.N * g.K / ms * 1e-9 );
	}
	// more FP16-out shapes from the command line: bench_tts_ops L reps N K [N K ...]
	for ( int i = 3; i + 1 < argc; i += 2 ) {
		const int N = std::atoi( argv[i] ), K = std::atoi( argv[i + 1] );
		ENGINE_ASSERT( static_cast<size_t>( N ) * K <= w_elems && K <= F && N <= 6 * F, "bench_tts_ops: shape beyond the buffers" );
		const double ms = time_ms( [&] { TtsOps::gemm( x, w, c, L, N, K, false ); }, reps );
		std::printf( "  extra %5d x %5d x %5d f16: %8.3f ms  %6.2f TFLOP/s\n", L, N, K, ms, 2.0 * L * N * K / ms * 1e-9 );
	}
	// one row's attention (the whole pass taken as one row)
	auto* q = static_cast<backend::half_t*>( A( static_cast<size_t>( L ) * d * 2 ) );
	auto* o = static_cast<backend::half_t*>( A( static_cast<size_t>( L ) * d * 2 ) );
	const size_t sb = TtsOps::attention_scratch( H, hd, L );
	void* scratch = sb ? A( sb ) : nullptr;
	const double ams = time_ms( [&] { TtsOps::attention( q, q, q, o, 0, L, L, H, hd, 0.125f, scratch, sb ); }, reps );
	std::printf( "attention %d positions x %d heads x %d: %8.3f ms  %6.2f TFLOP/s\n", L, H, hd, ams, 4.0 * L * L * H * hd / ams * 1e-9 );
	// codec convolutions: the decoder's first residual layer (768 channels at 2 x frames) and the last (24 channels at 512 x frames)
	const int frames = 400;
	const size_t chunk = 16u << 20;
	auto* cols = static_cast<float*>( A( chunk * 4 ) );
	auto* xin = static_cast<float*>( A( static_cast<size_t>( 64 ) * frames * 512 * 4 ) );
	auto* yout = static_cast<float*>( A( static_cast<size_t>( 64 ) * frames * 512 * 4 ) );
	auto* cw = static_cast<float*>( A( static_cast<size_t>( 768 ) * 768 * 11 * 4 ) );
	auto* cb = static_cast<float*>( A( 768 * 4 ) );
	struct Cv { int C, k, T; };
	const Cv convs[] = { { 384, 3, frames * 8 }, { 384, 11, frames * 8 }, { 48, 7, frames * 256 }, { 24, 11, frames * 512 } };
	for ( const Cv& v : convs ) {
		const double ms = time_ms( [&] { CodecOps::conv( xin, cw, cb, yout, v.C, v.C, v.k, 1, 1, ( v.k - 1 ) / 2, v.T, v.T, cols, chunk ); }, reps );
		std::printf( "conv %4d -> %4d, k %2d, T %7d: %8.3f ms  %6.2f TFLOP/s\n", v.C, v.C, v.k, v.T, ms, 2.0 * v.C * v.C * v.k * v.T / ms * 1e-9 );
	}
	engine::core::arena_destroy( arena );
	return 0;
}
