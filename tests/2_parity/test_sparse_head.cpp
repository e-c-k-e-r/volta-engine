// Sparse drafter head (modules/sparse_head) on one rank's real Flash-Next head shard: the columns read through identity blocks (SparseHead::extract)
// give the same logits as the full head at those columns (FP16 rounding of the weights only), every other column of the scattered row is the
// lowest value, and the subset's argmax is the full head's argmax restricted to the subset. Also times one extraction. One GPU:
// CUDA_VISIBLE_DEVICES=1 test_sparse_head [model dir]
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/modules/sparse_head.h"
#include "engine/runtime/test_paths.h"
#include "engine/spec/dflash.h"
#include "exl3_gemm.cuh"
#include "trellis_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

using engine::modules::SparseHead;

static uint32_t g_seed = 4242u;
static uint32_t rnd32() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

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
	const int tp = 4, H = 2560, vocab = 248320, real = vocab / tp, cols = real + ( 256 - real % 256 ) % 256, rows = 4, ident_rows = 128;
	const pod::LinearWeight head = repo.load_linear_slice_padded( "lm_head", arena, 6, 0, tp, cols - real, s );
	ENGINE_ASSERT( head.is_valid(), "lm_head shard not loaded" );

	auto alloc = [&]( size_t bytes ) { void* p = engine::core::arena_alloc( arena, bytes ); ENGINE_ASSERT( p, "arena" ); return p; };
	half* x = static_cast<half*>( alloc( static_cast<size_t>( rows ) * H * sizeof( half ) ) );
	half* full = static_cast<half*>( alloc( static_cast<size_t>( rows ) * cols * sizeof( half ) ) );
	half* sparse = static_cast<half*>( alloc( static_cast<size_t>( rows ) * cols * sizeof( half ) ) );
	std::vector<half> hx( static_cast<size_t>( rows ) * H );
	for ( auto& v : hx ) v = __float2half( ( static_cast<float>( rnd32() % 2000 ) / 1000.0f - 1.0f ) * 2.0f );
	CUDA_CHECK( cudaMemcpy( x, hx.data(), hx.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
	pod::TensorView tx = pod::make_tensor_2d( x, pod::DType::FP16, rows, H ), tf = pod::make_tensor_2d( full, pod::DType::FP16, rows, cols );
	engine::modules::Linear::forward( tx, head, tf, s );
	std::vector<half> hf( static_cast<size_t>( rows ) * cols );
	CUDA_CHECK( cudaMemcpy( hf.data(), full, hf.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );

	// subset: the 32 largest columns of every row and 4000 random ones (some rows' winners deliberately left out below)
	std::set<int> pick;
	for ( int r = 0; r < rows; ++r ) {
		std::vector<int> idx( real );
		for ( int c = 0; c < real; ++c ) idx[c] = c;
		std::partial_sort( idx.begin(), idx.begin() + 32, idx.end(), [&]( int a, int b ) { return __half2float( hf[static_cast<size_t>( r ) * cols + a] ) > __half2float( hf[static_cast<size_t>( r ) * cols + b] ); } );
		for ( int i = r == rows - 1 ? 1 : 0; i < 32; ++i ) pick.insert( idx[i] );   // the last row's winner is not in the subset
	}
	while ( pick.size() < 4096 ) pick.insert( static_cast<int>( rnd32() % real ) );
	std::vector<int> sel( pick.begin(), pick.end() );
	const int count = static_cast<int>( sel.size() );
	int* d_cols = static_cast<int*>( alloc( count * sizeof( int ) ) );
	half* w_sub = static_cast<half*>( alloc( static_cast<size_t>( count ) * H * sizeof( half ) ) );
	half* sub = static_cast<half*>( alloc( static_cast<size_t>( rows ) * count * sizeof( half ) ) );
	half* ident = static_cast<half*>( alloc( static_cast<size_t>( ident_rows ) * H * sizeof( half ) ) );
	half* chunk = static_cast<half*>( alloc( static_cast<size_t>( ident_rows ) * cols * sizeof( half ) ) );
	CUDA_CHECK( cudaMemcpy( d_cols, sel.data(), count * sizeof( int ), cudaMemcpyHostToDevice ) );
	const auto t0 = std::chrono::steady_clock::now();
	SparseHead::extract( head, H, cols, d_cols, count, ident, chunk, ident_rows, w_sub, s );
	const double ext_ms = std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count();
	engine::spec::dflash_f16_gemv_nt( w_sub, x, sub, rows, count, H, s );
	SparseHead::scatter( sub, count, d_cols, sparse, cols, rows, s );
	std::vector<half> hs( static_cast<size_t>( rows ) * cols );
	CUDA_CHECK( cudaMemcpy( hs.data(), sparse, hs.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );

	bool pass = true;
	for ( int r = 0; r < rows; ++r ) {
		const half* f = &hf[static_cast<size_t>( r ) * cols];
		const half* p = &hs[static_cast<size_t>( r ) * cols];
		double worst = 0.0, scale = 0.0;
		int outside_bad = 0, best_full = -1, best_sub = -1, best_sparse = -1;
		float bf = -1e30f, bs = -1e30f, bp = -1e30f;
		for ( int c = 0; c < cols; ++c ) {
			const float fv = __half2float( f[c] ), pv = __half2float( p[c] );
			if ( c < real && fv > bf ) { bf = fv; best_full = c; }
			if ( c < real && pv > bp ) { bp = pv; best_sparse = c; }
			if ( pick.count( c ) ) {
				worst = std::max( worst, static_cast<double>( std::fabs( fv - pv ) ) );
				scale = std::max( scale, static_cast<double>( std::fabs( fv ) ) );
				if ( fv > bs ) { bs = fv; best_sub = c; }
			}
			else if ( pv != -65504.0f ) ++outside_bad;
		}
		const bool argmax_ok = best_sparse == best_sub || std::fabs( __half2float( f[best_sparse] ) - bs ) <= 0.0625f;   // a tie within FP16 rounding
		const bool ok = worst <= 0.02 * scale && outside_bad == 0 && argmax_ok;
		pass &= ok;
		std::printf( "  row %d: max |sparse - full| %.4f (of %.2f) over %d columns, others at the floor: %s, argmax full %d, subset %d, sparse %d%s\n", r, worst, scale, count,
			outside_bad ? "NO" : "yes", best_full, best_sub, best_sparse, ok ? "" : "  FAIL" );
	}
	std::printf( "  extraction of %d columns: %.1f ms\n", count, ext_ms );
	std::printf( "test_sparse_head: %s\n", pass ? "OK" : "FAIL" );
	return pass ? 0 : 1;
}
