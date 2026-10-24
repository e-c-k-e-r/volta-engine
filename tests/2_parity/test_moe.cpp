// MoE parity: persistent MoE::forward vs per-expert reference (Linear + SwiGLU + accumulate)
// on real GLM-5.3-Flash layer-3 routed experts (rank-0 partition, 72 experts).
#include "engine/runtime/test_paths.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/modules/moe.h"
#include "engine/modules/swiglu.h"
#include "trellis_context.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	// MOE_FLASH=1: Flash-Next's experts (ENGINE_TEST_FLASH_DIR; MOE_BITS gives their bit width, 2 for Qwen3.8-Flash-Next-exl3) instead of GLM's
	// MOE_DSV41=1: DeepSeek-V4.1-Flash's experts of bit width MOE_BITS (2 to 5; ENGINE_TEST_DSV41_DIR), from the layer that has the most of them
	const bool flash = std::getenv( "MOE_FLASH" ) != nullptr;
	const bool dsv41 = std::getenv( "MOE_DSV41" ) != nullptr;
	const int EBITS = std::getenv( "MOE_BITS" ) ? std::atoi( std::getenv( "MOE_BITS" ) ) : 4;
	// MOE_SIG=g,u,d (with MOE_DSV41): experts whose gate / up / down projections have those widths (the mixed-width packs), instead of MOE_BITS for all three
	int sig[3] = { EBITS, EBITS, EBITS };
	if ( const char* sg = std::getenv( "MOE_SIG" ) ) std::sscanf( sg, "%d,%d,%d", &sig[0], &sig[1], &sig[2] );
	const char* dir = dsv41 ? engine::runtime::test::model_path( "DeepSeek-V4.1-Flash-1017-EXL3-2.7bpw", "ENGINE_TEST_DSV41_DIR" )
		: flash ? engine::runtime::test::model_path( "Swift-Qwen3.8-Flash-Next-exl3-4.00bpw", "ENGINE_TEST_FLASH_DIR" ) : engine::runtime::test::model_path( "GLM-5.3-Flash-exl3-3.05bpw", "ENGINE_TEST_GLM_DIR" );
	const int H = dsv41 ? 5120 : flash ? 2560 : 4096, I = dsv41 ? 2304 : flash ? 640 : 2048, K = dsv41 ? 6 : flash ? 10 : 8;
	int E = flash ? 128 : 72, L = std::getenv( "MOE_LAYER" ) ? std::atoi( std::getenv( "MOE_LAYER" ) ) : 3;   // MOE_LAYER: the layer of the checkpoint whose experts are tested
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	engine::modules::trellis_get_workspace( 0, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( 0 );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 2ULL * 1024 * 1024 * 1024 ), "arena" );

	engine::loader::SafeTensorsRepo repo;
	ENGINE_ASSERT( std::getenv( "MOE_SYNTH" ) || repo.open( dir ), "open model dir" );
	pod::MoELayerWeights mw;
	mw.type = pod::QuantType::TRELLIS;
	mw.num_local_experts = E;
	mw.top_k = K;
	char name[256];
	std::vector<int> dsv_ids;   // DeepSeek: the checkpoint ids of the chosen width's experts in the chosen layer
	if ( dsv41 ) {
		size_t best = 0;
		for ( int l = 0; l < 40; ++l ) {
			std::vector<int> ids;
			for ( int e = 0; e < 384; ++e ) {
				bool ok = true;
				const char* const ws[3] = { "w1", "w3", "w2" };
				for ( int q = 0; q < 3; ++q ) {
					std::snprintf( name, sizeof( name ), "layers.%d.ffn.experts.%d.%s.trellis", l, e, ws[q] );
					const auto* m = repo.find( name );
					ok = ok && m != nullptr && m->shape[2] / 16 == sig[q];
				}
				if ( ok && ids.size() < 96 ) ids.push_back( e );
			}
			if ( ids.size() > best ) { best = ids.size(); dsv_ids = ids; L = l; }
		}
		ENGINE_ASSERT( !dsv_ids.empty(), "test_moe: no expert of that width" );
		E = static_cast<int>( dsv_ids.size() );
		mw.num_local_experts = E;
		std::printf( "[*] DeepSeek-V4.1: %d experts of %d/%d/%d bits in layer %d\n", E, sig[0], sig[1], sig[2], L );
	}
	// MOE_SYNTH=1: random trellis words and small random signs instead of the checkpoint's experts, in whatever tile order the build reads
	// (ENGINE_EXL3_VOLTA_LAYOUT builds cannot read a stock checkpoint); MOE_SIG=g,u,d sets the widths of the three projections
	const bool synth = std::getenv( "MOE_SYNTH" ) != nullptr;
	uint32_t synth_state = 777u;
	auto synth_linear = [&]( int k, int n, int bits ) {
		const size_t words = static_cast<size_t>( k / 16 ) * ( n / 16 ) * 16 * bits;
		std::vector<uint16_t> ht( words );
		for ( auto& w : ht ) { synth_state = synth_state * 1664525u + 1013904223u; w = static_cast<uint16_t>( synth_state >> 16 ); }
		std::vector<half> hu( k ), hv( n );
		for ( auto& v : hu ) { synth_state = synth_state * 1664525u + 1013904223u; v = __float2half( ( synth_state >> 16 ) & 1 ? 1.0f : -1.0f ); }
		for ( auto& v : hv ) { synth_state = synth_state * 1664525u + 1013904223u; v = __float2half( ( ( synth_state >> 16 ) & 1 ? 1.0f : -1.0f ) * 0.02f ); }
		int tshape[3] = { k / 16, n / 16, 16 * bits };
		pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, tshape, 3 );
		int ks[1] = { k }, ns[1] = { n };
		pod::TensorView su = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ks, 1 );
		pod::TensorView sv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ns, 1 );
		CUDA_CHECK( cudaMemcpy( tv.data, ht.data(), words * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( su.data, hu.data(), k * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( sv.data, hv.data(), n * 2, cudaMemcpyHostToDevice ) );
		return pod::LinearWeight::make_trellis( tv, su, sv, bits, true, false );
	};
	for ( int e = 0; e < E; ++e ) {
		if ( synth ) {
			mw.gate[e] = synth_linear( H, I, sig[0] );
			mw.up[e] = synth_linear( H, I, sig[1] );
			mw.down[e] = synth_linear( I, H, sig[2] );
			continue;
		}
		if ( dsv41 ) {
			const char* parts[3] = { "w1", "w3", "w2" };
			pod::LinearWeight* dst[3] = { &mw.gate[e], &mw.up[e], &mw.down[e] };
			for ( int p = 0; p < 3; ++p ) {
				std::snprintf( name, sizeof( name ), "layers.%d.ffn.experts.%d.%s", L, dsv_ids[e], parts[p] );
				*dst[p] = repo.load_linear( name, arena, sig[p], stream );
			}
			continue;
		}
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, e );
		mw.gate[e] = repo.load_linear( name, arena, EBITS, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, e );
		mw.up[e] = repo.load_linear( name, arena, EBITS, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, e );
		mw.down[e] = repo.load_linear( name, arena, EBITS, stream );
	}
	engine::modules::MoE::init_device_ptrs( mw, arena, stream );
	std::printf( "[*] experts: bits g/u/d = %d/%d/%d, mul1 g/u/d = %d/%d/%d\n",
		mw.gate[0].bits, mw.up[0].bits, mw.down[0].bits, mw.gate[0].mul1, mw.up[0].mul1, mw.down[0].mul1 );

	// Inputs: x ~ U(-1, 1); routes include out-of-partition experts (>= 72) that must be skipped
	int h_idx[10] = { 3, 17, 40, 71, 100, 5, 250, 66 };
	if ( dsv41 ) for ( int k = 0; k < K; ++k ) h_idx[k] = k * 7 < E ? k * 7 : E + k;   // in-range ids where the group is large enough
	if ( flash ) { const int fi[10] = { 3, 17, 40, 71, 127, 5, 250, 66, 9, 300 }; for ( int k = 0; k < K; ++k ) h_idx[k] = fi[k]; }
	// MOE_ACTIVE=n keeps only the first n routes (others moved out of partition) for bisection
	if ( const char* env = std::getenv( "MOE_ACTIVE" ) ) {
		for ( int k = std::atoi( env ); k < K; ++k ) h_idx[k] = 300;
	}
	const float h_wf[10] = { 0.21f, 0.17f, 0.14f, 0.12f, 0.11f, 0.10f, 0.08f, 0.07f, 0.05f, 0.04f };
	std::vector<half> h_x( H ), h_w( K );
	uint32_t seed = 12345u;
	for ( int i = 0; i < H; ++i ) {
		seed = seed * 1664525u + 1013904223u;
		h_x[i] = __float2half( ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f );
	}
	for ( int k = 0; k < K; ++k ) h_w[k] = __float2half( h_wf[k] );

	int hs[2] = { 1, H }, is[2] = { 1, I };
	auto x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, hs, 2 );
	auto ref = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, hs, 2 );
	auto out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, hs, 2 );
	auto eo = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, hs, 2 );
	auto g = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, is, 2 );
	auto u = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, is, 2 );
	auto inter = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, is, 2 );
	int* d_idx = static_cast<int*>( engine::core::arena_alloc( arena, K * sizeof( int ) ) );
	half* d_w = static_cast<half*>( engine::core::arena_alloc( arena, K * sizeof( half ) ) );
	CUDA_CHECK( cudaMemcpy( x.data, h_x.data(), H * sizeof( half ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_idx, h_idx, sizeof( h_idx ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_w, h_w.data(), K * sizeof( half ), cudaMemcpyHostToDevice ) );

	// Reference: per-expert Linear path (the validated pre-refactor decode path)
	CUDA_CHECK( cudaMemset( ref.data, 0, ref.size_bytes() ) );
	for ( int k = 0; k < K; ++k ) {
		if ( h_idx[k] >= E ) continue;
		const int e = h_idx[k];
		engine::modules::Linear::forward( x, mw.gate[e], g, stream );
		engine::modules::Linear::forward( x, mw.up[e], u, stream );
		engine::modules::SwiGLU::forward( g, u, inter, stream, 10.0f );
		engine::modules::Linear::forward( inter, mw.down[e], eo, stream );
		engine::modules::MoERouter::accumulate( ref.ptr<half>(), eo.ptr<const half>(), h_w[k], H, stream );
	}

	auto scratch = engine::modules::MoE::alloc_scratch( arena, E, K, H, I );
	engine::modules::MoE::forward( x, mw, d_idx, d_w, out, 0, E, scratch, 10.0f, stream );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	std::vector<half> h_ref( H ), h_out( H );
	CUDA_CHECK( cudaMemcpy( h_ref.data(), ref.data, H * sizeof( half ), cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( h_out.data(), out.data, H * sizeof( half ), cudaMemcpyDeviceToHost ) );
	double dot = 0, na = 0, nb = 0, maxd = 0, maxr = 0;
	for ( int i = 0; i < H; ++i ) {
		const double a = __half2float( h_ref[i] ), b = __half2float( h_out[i] );
		dot += a * b;
		na += a * a;
		nb += b * b;
		maxd = std::fmax( maxd, std::fabs( a - b ) );
		maxr = std::fmax( maxr, std::fabs( a ) );
	}
	const double cos = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-12 );
	std::printf( "[*] MoE::forward vs reference: cos=%.6f max_abs=%.4f (ref max %.4f, |ref|=%.3f |out|=%.3f)\n",
		cos, maxd, maxr, std::sqrt( na ), std::sqrt( nb ) );
	const bool pass = cos >= 0.9999;
	std::printf( "[%c] MoE PARITY %s\n", pass ? '+' : '-', pass ? "PASS" : "FAIL" );

	// Multi-row: R rows routed into a narrow expert set so several experts exceed the 8-token chunk. Reference is
	// the single-row MoE::forward per row (the validated path).
	bool multi_pass = true;
	std::vector<int> r_list = { 2, 3, 5, 8, 9, 24, 64 };
	if ( const char* env = std::getenv( "MOE_ROWS" ) ) {
		r_list.clear();
		for ( const char* p = env; *p; ) {
			r_list.push_back( std::atoi( p ) );
			while ( *p && *p != ',' ) ++p;
			if ( *p == ',' ) ++p;
		}
	}
	for ( int R : r_list ) {
		int rs[2] = { R, H };
		auto xr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rs, 2 );
		auto outr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rs, 2 );
		int* d_idx_r = static_cast<int*>( engine::core::arena_alloc( arena, R * K * sizeof( int ) ) );
		half* d_w_r = static_cast<half*>( engine::core::arena_alloc( arena, R * K * sizeof( half ) ) );
		std::vector<half> hxr( static_cast<size_t>( R ) * H ), hwr( R * K );
		std::vector<int> hir( R * K );
		for ( auto& v : hxr ) {
			seed = seed * 1664525u + 1013904223u;
			v = __float2half( ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f );
		}
		for ( int r = 0; r < R; ++r ) {
			int used[400] = {};
			for ( int k = 0; k < K; ++k ) {
				int e;
				do {
					seed = seed * 1664525u + 1013904223u;
					e = ( ( seed >> 12 ) % 4 == 0 ) ? E + ( ( seed >> 4 ) % 200 ) : ( ( seed >> 4 ) % ( flash ? 14 : dsv41 ? std::max( E, 1 ) : 10 ) );
				} while ( used[e] );
				used[e] = 1;
				hir[r * K + k] = e;
				hwr[r * K + k] = __float2half( h_wf[k] );
			}
		}
		CUDA_CHECK( cudaMemcpy( xr.data, hxr.data(), hxr.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_idx_r, hir.data(), hir.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_w_r, hwr.data(), hwr.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		auto scratch_r = engine::modules::MoE::alloc_scratch( arena, E, K, H, I, R );
		engine::modules::MoE::forward( xr, mw, d_idx_r, d_w_r, outr, 0, E, scratch_r, 10.0f, stream );
		std::vector<half> h_got( static_cast<size_t>( R ) * H ), h_want( static_cast<size_t>( R ) * H );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		CUDA_CHECK( cudaMemcpy( h_got.data(), outr.data, h_got.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		for ( int r = 0; r < R; ++r ) {
			pod::TensorView x1 = pod::make_tensor_2d( xr.ptr<half>() + static_cast<size_t>( r ) * H, pod::DType::FP16, 1, H );
			pod::TensorView o1 = pod::make_tensor_2d( outr.ptr<half>() + static_cast<size_t>( r ) * H, pod::DType::FP16, 1, H );
			engine::modules::MoE::forward( x1, mw, d_idx_r + r * K, d_w_r + r * K, o1, 0, E, scratch, 10.0f, stream );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			CUDA_CHECK( cudaMemcpy( h_want.data() + static_cast<size_t>( r ) * H, o1.data, H * sizeof( half ), cudaMemcpyDeviceToHost ) );
		}
		double dot = 0, na = 0, nb = 0, maxd = 0;
		for ( size_t i = 0; i < h_got.size(); ++i ) {
			const double a = __half2float( h_want[i] ), b = __half2float( h_got[i] );
			dot += a * b;
			na += a * a;
			nb += b * b;
			maxd = std::fmax( maxd, std::fabs( a - b ) );
		}
		const double c = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-12 );
		if ( std::getenv( "MOE_ROWS_VERBOSE" ) && R <= 9 ) {
			for ( int r = 0; r < R; ++r ) {
				double d2 = 0, a2 = 0, b2 = 0;
				for ( int i = 0; i < H; ++i ) {
					const double a = __half2float( h_want[static_cast<size_t>( r ) * H + i] ), b = __half2float( h_got[static_cast<size_t>( r ) * H + i] );
					d2 += a * b; a2 += a * a; b2 += b * b;
				}
				std::printf( "    row %d: cos %.6f routes:", r, d2 / ( std::sqrt( a2 * b2 ) + 1e-12 ) );
				for ( int k = 0; k < K; ++k ) std::printf( " %d", hir[r * K + k] );
				std::printf( "\n" );
			}
		}
		std::printf( "[*] MoE %2d rows vs per-row: cos=%.6f max_abs=%.5f (|ref|=%.2f)\n", R, c, maxd, std::sqrt( na ) );
		multi_pass = multi_pass && c >= 0.9999;
	}
	std::printf( "[%c] MoE MULTI-ROW %s\n", multi_pass ? '+' : '-', multi_pass ? "PASS" : "FAIL" );
	repo.close();
	engine::core::arena_destroy( arena );
	engine::modules::trellis_context_destroy();
	(void) cudaDeviceReset();
	return 0;
}
