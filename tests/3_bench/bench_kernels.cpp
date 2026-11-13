// Projection cost sweep: modules::Linear::forward on the models' real projection shapes, every weight format the engine multiplies
// (EXL3 in both tile orders at the widths the checkpoints use, NVFP4, GGUF block types), and every row count a step can have (1 to 16 for
// decode and speculative verify, 32 to 512 for prefill chunks). Random weights (timing only), each case rotating over enough weight copies
// to stay out of L2, best of --reps. Writes CSV (shape,format,rows,us) to stdout; tests/tools/kernel_cliffs.py flags the cells that cost
// more than the curve around them predicts.
//   bench_kernels [--reps 3] [--only SUBSTRING] [--rows 1,2,...]
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/modules/linear.h"
#include "engine/modules/gguf.h"
#include "engine/modules/trellis.h"
#include "trellis_context.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace engine;
using backend::half_t;

namespace {

struct Shape { const char* name; int k, n; };

// per-rank shapes of the served models at TP=4 (the column-parallel ones split n, the row-parallel ones k)
const Shape kShapes[] = {
	{ "27B q+gate", 5120, 3072 }, { "27B gdn qkv", 5120, 2560 }, { "27B o", 1536, 5120 }, { "27B mlp gate/up", 5120, 4352 }, { "27B mlp down", 4352, 5120 },
	{ "27B lm_head/4", 5120, 62080 },
	{ "flash attn/gdn in", 2560, 4096 }, { "flash out", 4096, 2560 }, { "flash shared gate/up", 2560, 1024 }, { "flash lm_head/4", 2560, 62080 },
	{ "glm kda in", 4096, 6144 }, { "glm k4096 n4096", 4096, 4096 }, { "glm k4096 n2048", 4096, 2048 }, { "glm k2048 n4096", 2048, 4096 },
	{ "drafter gate/up", 5120, 17408 }, { "drafter down", 17408, 5120 },
};

struct Format { std::string name; int kind; int bits; bool volta; modules::GgmlType gtype; };   // kind 0 EXL3, 1 NVFP4, 2 GGUF

pod::MemoryArena g_arena{};

pod::TensorView alloc( pod::DType dt, std::initializer_list<int> shape ) {
	int s[4] = {}; int nd = 0;
	for ( int d : shape ) s[nd++] = d;
	return core::arena_alloc_tensor( g_arena, dt, s, nd );
}

void fill_random( void* dev, size_t bytes, uint32_t seed ) {
	std::vector<uint8_t> h( bytes );
	std::mt19937 rng( seed );
	for ( auto& v : h ) v = static_cast<uint8_t>( rng() );
	backend::memcpy_sync( dev, h.data(), bytes, backend::Copy::H2D );
}

void fill_half( void* dev, size_t n, float v ) {
	std::vector<half_t> h( n, backend::from_float_half( v ) );
	backend::memcpy_sync( dev, h.data(), n * sizeof( half_t ), backend::Copy::H2D );
}

// One weight of the format at [k, n]; false when the format cannot hold the shape
bool make_weight( const Format& f, const Shape& sh, uint32_t seed, pod::LinearWeight* out ) {
	const int k = sh.k, n = sh.n;
	if ( f.kind == 0 ) {
		if ( k % 128 || n % 128 ) return false;
		auto codes = alloc( pod::DType::UINT16, { k / 16, n / 16, 16 * f.bits } );
		auto suh = alloc( pod::DType::FP16, { k } );
		auto svh = alloc( pod::DType::FP16, { n } );
		fill_random( codes.data, static_cast<size_t>( k / 16 ) * ( n / 16 ) * 16 * f.bits * 2, seed );
		fill_half( suh.data, k, 1.0f );
		fill_half( svh.data, n, 0.5f );
		*out = pod::LinearWeight::make_trellis( codes, suh, svh, f.bits, true, false );
		out->volta_tiles = f.volta;
		return true;
	}
	if ( f.kind == 1 ) {
		if ( k % 16 || n % 32 ) return false;
		const int G = k / 16, tiles = n / 32;
		auto codes = alloc( pod::DType::UINT8, { tiles, G, 256 } );
		auto scales = alloc( pod::DType::UINT8, { tiles, G, 32 } );
		fill_random( codes.data, static_cast<size_t>( tiles ) * G * 256, seed );
		std::vector<uint8_t> hs( static_cast<size_t>( tiles ) * G * 32 );
		std::mt19937 rng( seed + 1 );
		for ( auto& v : hs ) v = static_cast<uint8_t>( ( 6 + rng() % 3 ) << 3 );   // E4M3 exponents 6 to 8, positive, no NaN
		backend::memcpy_sync( scales.data, hs.data(), hs.size(), backend::Copy::H2D );
		*out = pod::LinearWeight::make_nvfp4( codes, scales, 0.0625f );
		return true;
	}
	const modules::GgmlBlock b = modules::ggml_block( f.gtype );
	if ( b.elems == 0 || k % b.elems ) return false;
	const size_t bytes = static_cast<size_t>( n ) * ( k / b.elems ) * b.bytes;
	std::vector<uint8_t> h( bytes );
	std::mt19937 rng( seed );
	for ( auto& v : h ) v = static_cast<uint8_t>( rng() );
	// the half scales small, so the products stay in FP16 range: Q6_K keeps its d at the end of the block, the others at the start (and dmin next)
	const half_t d = backend::from_float_half( 1e-4f );
	const int d_at = f.gtype == modules::GgmlType::Q6_K ? b.bytes - 2 : 0;
	for ( size_t o = 0; o < bytes; o += b.bytes ) {
		std::memcpy( &h[o + d_at], &d, 2 );
		if ( f.gtype == modules::GgmlType::Q4_K || f.gtype == modules::GgmlType::Q5_K ) std::memcpy( &h[o + 2], &d, 2 );
	}
	auto blocks = alloc( pod::DType::UINT8, { static_cast<int>( bytes ) } );
	backend::memcpy_sync( blocks.data, h.data(), bytes, backend::Copy::H2D );
	*out = pod::LinearWeight::make_gguf( blocks, static_cast<uint32_t>( f.gtype ), k, n );
	return true;
}

size_t weight_bytes( const Format& f, const Shape& sh ) {
	if ( f.kind == 0 ) return static_cast<size_t>( sh.k ) * sh.n * f.bits / 8;
	if ( f.kind == 1 ) return static_cast<size_t>( sh.k ) * sh.n / 2 + static_cast<size_t>( sh.k ) * sh.n / 16;
	const modules::GgmlBlock b = modules::ggml_block( f.gtype );
	return b.elems ? static_cast<size_t>( sh.n ) * ( sh.k / b.elems ) * b.bytes : 0;
}

} // namespace

int main( int argc, char** argv ) {
	int reps = 3;
	std::string only;
	std::vector<int> rows = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512 };
	for ( int a = 1; a < argc; ++a ) {
		if ( !std::strcmp( argv[a], "--reps" ) && a + 1 < argc ) reps = std::atoi( argv[++a] );
		else if ( !std::strcmp( argv[a], "--only" ) && a + 1 < argc ) only = argv[++a];
		else if ( !std::strcmp( argv[a], "--rows" ) && a + 1 < argc ) {
			rows.clear();
			for ( const char* q = argv[++a]; *q; ) { rows.push_back( std::atoi( q ) ); while ( *q && *q != ',' ) ++q; if ( *q ) ++q; }
		}
	}
	backend::set_device( 0 );
	modules::trellis_context_init_all();
	ENGINE_ASSERT( core::arena_create( g_arena, 0, 12ull << 30 ), "arena" );
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );

	std::vector<Format> formats;
	for ( int b : { 3, 4, 5 } ) {
		formats.push_back( { "exl3 " + std::to_string( b ) + "b stock", 0, b, false, modules::GgmlType::F32 } );
		if ( modules::trellis_volta_tiles_supported() ) formats.push_back( { "exl3 " + std::to_string( b ) + "b volta", 0, b, true, modules::GgmlType::F32 } );
	}
	formats.push_back( { "nvfp4", 1, 4, false, modules::GgmlType::F32 } );
	formats.push_back( { "gguf q4_k", 2, 0, false, modules::GgmlType::Q4_K } );
	formats.push_back( { "gguf q6_k", 2, 0, false, modules::GgmlType::Q6_K } );
	formats.push_back( { "gguf iq3_s", 2, 0, false, modules::GgmlType::IQ3_S } );

	const int mmax = *std::max_element( rows.begin(), rows.end() );
	std::printf( "shape,format,rows,us\n" );
	for ( const Shape& sh : kShapes ) {
		if ( !only.empty() && std::string( sh.name ).find( only ) == std::string::npos ) continue;
		auto x = alloc( pod::DType::FP16, { mmax, sh.k } );
		auto y = alloc( pod::DType::FP16, { mmax, sh.n } );
		fill_half( x.data, static_cast<size_t>( mmax ) * sh.k, 0.01f );
		for ( const Format& f : formats ) {
			const size_t wb = weight_bytes( f, sh );
			if ( !wb ) continue;
			const int copies = static_cast<int>( std::max<size_t>( 2, std::min<size_t>( 16, ( 48ull << 20 ) / wb + 1 ) ) );
			const size_t mark = g_arena.offset_bytes;
			std::vector<pod::LinearWeight> w( copies );
			bool ok = true;
			for ( int c = 0; c < copies && ok; ++c ) ok = make_weight( f, sh, 1234u + c, &w[c] );
			if ( !ok ) { g_arena.offset_bytes = mark; continue; }
			for ( int m : rows ) {
				pod::TensorView xt = pod::make_tensor_2d( x.data, pod::DType::FP16, m, sh.k );
				pod::TensorView yt = pod::make_tensor_2d( y.data, pod::DType::FP16, m, sh.n );
				for ( int c = 0; c < copies; ++c ) modules::Linear::forward( xt, w[c], yt, s );   // warm-up (also builds any launch caches)
				double best = 1e30;
				const int iters = std::max( 2 * copies, 8 );
				for ( int r = 0; r < reps; ++r ) {
					cudaEvent_t a, b;
					CUDA_CHECK( cudaEventCreate( &a ) );
					CUDA_CHECK( cudaEventCreate( &b ) );
					CUDA_CHECK( cudaEventRecord( a, s ) );
					for ( int i = 0; i < iters; ++i ) modules::Linear::forward( xt, w[i % copies], yt, s );
					CUDA_CHECK( cudaEventRecord( b, s ) );
					CUDA_CHECK( cudaEventSynchronize( b ) );
					float ms = 0;
					CUDA_CHECK( cudaEventElapsedTime( &ms, a, b ) );
					best = std::min( best, 1000.0 * ms / iters );
					cudaEventDestroy( a );
					cudaEventDestroy( b );
				}
				std::printf( "%s,%s,%d,%.2f\n", sh.name, f.name.c_str(), m, best );
				std::fflush( stdout );
			}
			g_arena.offset_bytes = mark;
		}
	}
	return 0;
}
