// MoE parity: persistent MoE::forward vs per-expert reference (Linear + SwiGLU + accumulate)
// on real GLM-5.3-Flash layer-3 routed experts (rank-0 partition, 72 experts).
#include "engine/modules/trellis_bits.h"
#include "engine/runtime/test_paths.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/modules/moe.h"
#include "engine/modules/swiglu.h"
#include "trellis_context.h"
#include "reconstruct.cuh"
#include "exl3_vl.cuh"
#include "exl3_moe_coop.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>

// the tile orders a build reads: both on SM_70 CUDA builds (ENGINE_EXL3_VL), else the stock one
#if ENGINE_EXL3_VL
static constexpr int kVoltaOrders = 2;
#else
static constexpr int kVoltaOrders = 1;
#endif

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	// MOE_FLASH=1: Flash-Next's experts (ENGINE_TEST_FLASH_DIR; MOE_BITS gives their bit width, 2 for Qwen3.8-Flash-Next-exl3) instead of GLM's
	// MOE_DSV41=1: DeepSeek-V4.1-Flash's experts of width code MOE_BITS (1 to 5, 9 for 1.5 bpw; ENGINE_TEST_DSV41_DIR), from the layer that has the most of them
	const bool flash = std::getenv( "MOE_FLASH" ) != nullptr;
	const bool dsv41 = std::getenv( "MOE_DSV41" ) != nullptr;
	const int EBITS = std::getenv( "MOE_BITS" ) ? std::atoi( std::getenv( "MOE_BITS" ) ) : 4;
	// MOE_SIG=g,u,d (with MOE_DSV41): experts whose gate / up / down projections have those widths (the mixed-width packs), instead of MOE_BITS for all three
	int sig[3] = { EBITS, EBITS, EBITS };
	if ( const char* sg = std::getenv( "MOE_SIG" ) ) std::sscanf( sg, "%d,%d,%d", &sig[0], &sig[1], &sig[2] );
	const char* dir = dsv41 ? engine::runtime::test::model_path( "ENGINE_TEST_DSV41_DIR" )
		: flash ? engine::runtime::test::model_path( "ENGINE_TEST_FLASH_DIR" ) : engine::runtime::test::model_path( "ENGINE_TEST_GLM_DIR" );
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
					ok = ok && m != nullptr && engine::modules::trellis_code_from_u16( m->shape[2] ) == sig[q];
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
	// (ENGINE_EXL3_VOLTA_LAYOUT builds cannot read a stock checkpoint); MOE_SIG=g,u,d sets the widths of the three projections (width codes: 10 and 11 are 2.5 and 3.5 bpw)
	const bool synth = std::getenv( "MOE_SYNTH" ) != nullptr;
	uint32_t synth_state = 777u;
	auto synth_linear = [&]( int k, int n, int bits ) {
		const size_t words = static_cast<size_t>( k / 16 ) * ( n / 16 ) * engine::modules::trellis_tile_u16( bits );
		std::vector<uint16_t> ht( words );
		for ( auto& w : ht ) { synth_state = synth_state * 1664525u + 1013904223u; w = static_cast<uint16_t>( synth_state >> 16 ); }
		std::vector<half> hu( k ), hv( n );
		for ( auto& v : hu ) { synth_state = synth_state * 1664525u + 1013904223u; v = __float2half( ( synth_state >> 16 ) & 1 ? 1.0f : -1.0f ); }
		for ( auto& v : hv ) { synth_state = synth_state * 1664525u + 1013904223u; v = __float2half( ( ( synth_state >> 16 ) & 1 ? 1.0f : -1.0f ) * 0.02f ); }
		int tshape[3] = { k / 16, n / 16, engine::modules::trellis_tile_u16( bits ) };
		pod::TensorView tv = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, tshape, 3 );
		int ks[1] = { k }, ns[1] = { n };
		pod::TensorView su = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ks, 1 );
		pod::TensorView sv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ns, 1 );
		CUDA_CHECK( cudaMemcpy( tv.data, ht.data(), words * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( su.data, hu.data(), k * 2, cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( sv.data, hv.data(), n * 2, cudaMemcpyHostToDevice ) );
		return pod::LinearWeight::make_trellis( tv, su, sv, bits, true, false );
	};
	// MOE_FIRST=n (Flash-Next): the experts from id n on (another rank's partition)
	const int first = std::getenv( "MOE_FIRST" ) ? std::atoi( std::getenv( "MOE_FIRST" ) ) : 0;
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
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, e + first );
		mw.gate[e] = repo.load_linear( name, arena, EBITS, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, e + first );
		mw.up[e] = repo.load_linear( name, arena, EBITS, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, e + first );
		mw.down[e] = repo.load_linear( name, arena, EBITS, stream );
	}
	engine::modules::MoE::init_device_ptrs( mw, arena, stream );
	std::printf( "[*] experts: bits g/u/d = %d/%d/%d, mul1 g/u/d = %d/%d/%d\n",
		mw.gate[0].bits, mw.up[0].bits, mw.down[0].bits, mw.gate[0].mul1, mw.up[0].mul1, mw.down[0].mul1 );
	// A checkpoint of the extended trellis (tile starts / gains): Linear::forward refuses it, so the single-row parity is left out and the
	// MOE_EXT checks run on the checkpoint's own experts and tile order
	const bool ext_real = !synth && mw.ext_ptrs != nullptr;
	if ( ext_real ) std::printf( "[*] extended trellis experts (%s tile order)\n", mw.gate[0].volta_tiles ? "Volta" : "stock" );

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

	auto scratch = engine::modules::MoE::alloc_scratch( arena, E, K, H, I );
	if ( !ext_real ) {
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
	}

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

	// MOE_EXT=1 (with MOE_SYNTH): the extended trellis, random tile starts and gains on every expert, in both tile orders: the coop kernels at 1 to
	// 16 rows against a host product with the weights reconstruct_had_slice_ext gives (the quantizer's decode), the persistent kernel's 4-, 16- and
	// 32-row tiles at 24 to 200 rows against those rows one at a time; the same routes with the companions dropped must miss
	if ( ( synth && std::getenv( "MOE_EXT" ) ) || ext_real ) {
		bool ext_pass = true;
		std::mt19937 rng( 4242u );
		auto attach = [&]( pod::LinearWeight& lw ) {
			const int k = lw.w1.shape[0], n = lw.w2.shape[0], tiles = ( k / 16 ) * ( n / 16 );
			const int d0 = lw.bits > 8 ? lw.bits - 8 : lw.bits;   // the start holds the 16 - d0 bits before the stream
			std::vector<uint16_t> st( tiles );
			std::vector<uint8_t> gn( tiles );
			for ( auto& s : st ) s = static_cast<uint16_t>( rng() & ( ( 1u << ( 16 - d0 ) ) - 1 ) );
			for ( auto& g : gn ) g = static_cast<uint8_t>( rng() & 255u );
			int shp[2] = { k / 16, n / 16 };
			lw.w3 = engine::core::arena_alloc_tensor( arena, pod::DType::UINT16, shp, 2 );
			lw.w4 = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, shp, 2 );
			CUDA_CHECK( cudaMemcpy( lw.w3.data, st.data(), st.size() * 2, cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( lw.w4.data, gn.data(), gn.size(), cudaMemcpyHostToDevice ) );
		};
		if ( synth ) {
			for ( int e = 0; e < E; ++e ) { attach( mw.gate[e] ); attach( mw.up[e] ); attach( mw.down[e] ); }
			engine::modules::MoE::init_device_ptrs( mw, arena, stream );
		}
		ENGINE_ASSERT( mw.ext_ptrs != nullptr, "test_moe: no companion table" );
		const int NE = 10;   // routes go to experts 0 .. NE - 1 or out of the partition
		half* d_dense = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( H ) * I * sizeof( half ) ) );
		const int R = 200, RC = MOE_COOP_MAX_ROWS;   // rows in all, rows of the host reference
		int rs[2] = { R, H };
		auto xr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rs, 2 );
		auto outr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rs, 2 );
		int* d_idx_r = static_cast<int*>( engine::core::arena_alloc( arena, R * K * sizeof( int ) ) );
		half* d_w_r = static_cast<half*>( engine::core::arena_alloc( arena, R * K * sizeof( half ) ) );
		std::vector<half> hxr( static_cast<size_t>( R ) * H ), hwr( R * K );
		std::vector<int> hir( R * K );
		for ( auto& v : hxr ) v = __float2half( std::uniform_real_distribution<float>( -1.0f, 1.0f )( rng ) );
		for ( int r = 0; r < R; ++r ) {
			int used[400] = {};
			for ( int k = 0; k < K; ++k ) {
				int e;
				do e = rng() % 4 == 0 ? E + static_cast<int>( rng() % 200 ) : static_cast<int>( rng() % NE ); while ( used[e] );
				used[e] = 1;
				hir[r * K + k] = e;
				hwr[r * K + k] = __float2half( h_wf[k] );
			}
		}
		CUDA_CHECK( cudaMemcpy( xr.data, hxr.data(), hxr.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_idx_r, hir.data(), hir.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_w_r, hwr.data(), hwr.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		auto scratch_r = engine::modules::MoE::alloc_scratch( arena, E, K, H, I, R );
		for ( int vt = ext_real ? static_cast<int>( mw.gate[0].volta_tiles ) : 0; vt < ( ext_real ? static_cast<int>( mw.gate[0].volta_tiles ) + 1 : kVoltaOrders ); ++vt ) {
			if ( !ext_real ) for ( int e = 0; e < E; ++e ) mw.gate[e].volta_tiles = mw.up[e].volta_tiles = mw.down[e].volta_tiles = vt == 1;
			// the experts' dense weights, [k, n] row-major
			auto dense = [&]( const pod::LinearWeight& lw ) {
				const int k = lw.w1.shape[0], n = lw.w2.shape[0];
				EXL3_TILES_CALL( lw.volta_tiles, reconstruct_had_slice_ext, d_dense, lw.w0.ptr<const uint16_t>(), lw.w3.ptr<const uint16_t>(), lw.w4.ptr<const uint8_t>(),
					lw.w1.ptr<const half>(), lw.w2.ptr<const half>(), lw.bits, lw.mcg, lw.mul1, k, n, 0, stream );
				std::vector<half> hw( static_cast<size_t>( k ) * n );
				CUDA_CHECK( cudaMemcpyAsync( hw.data(), d_dense, hw.size() * sizeof( half ), cudaMemcpyDeviceToHost, stream ) );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				std::vector<float> f( hw.size() );
				for ( size_t i = 0; i < f.size(); ++i ) f[i] = __half2float( hw[i] );
				return f;
			};
			std::vector<std::vector<float>> wg( NE ), wu( NE ), wd( NE );
			for ( int e = 0; e < NE; ++e ) { wg[e] = dense( mw.gate[e] ); wu[e] = dense( mw.up[e] ); wd[e] = dense( mw.down[e] ); }
			// FP16 roundings where the kernels store FP16: the gate / up outputs, the activation, the down output
			auto r16 = []( float v ) { return __half2float( __float2half( v ) ); };
			auto reference = [&]( int r, std::vector<float>& o ) {
				o.assign( H, 0.0f );
				std::vector<float> g( I ), u( I ), a( I ), d( H );
				for ( int k = 0; k < K; ++k ) {
					const int e = hir[r * K + k];
					if ( e >= E ) continue;
					std::fill( g.begin(), g.end(), 0.0f );
					std::fill( u.begin(), u.end(), 0.0f );
					for ( int h = 0; h < H; ++h ) {
						const float xv = __half2float( hxr[static_cast<size_t>( r ) * H + h] );
						const float* pg = wg[e].data() + static_cast<size_t>( h ) * I;
						const float* pu = wu[e].data() + static_cast<size_t>( h ) * I;
						for ( int i = 0; i < I; ++i ) { g[i] += xv * pg[i]; u[i] += xv * pu[i]; }
					}
					for ( int i = 0; i < I; ++i ) {
						const float gv = std::fmin( r16( g[i] ), 10.0f ), uv = std::fmax( -10.0f, std::fmin( r16( u[i] ), 10.0f ) );
						a[i] = r16( gv / ( 1.0f + std::exp( -gv ) ) * uv );
					}
					std::fill( d.begin(), d.end(), 0.0f );
					for ( int i = 0; i < I; ++i ) {
						const float* pd = wd[e].data() + static_cast<size_t>( i ) * H;
						for ( int h = 0; h < H; ++h ) d[h] += a[i] * pd[h];
					}
					for ( int h = 0; h < H; ++h ) o[h] += __half2float( hwr[r * K + k] ) * r16( d[h] );
				}
			};
			std::vector<std::vector<float>> want( RC );
			for ( int r = 0; r < RC; ++r ) reference( r, want[r] );
			auto cosine = [&]( const std::vector<half>& got, int rows ) {
				double dot = 0, na = 0, nb = 0;
				for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < H; ++h ) {
					const double a = want[r][h], b = __half2float( got[static_cast<size_t>( r ) * H + h] );
					dot += a * b; na += a * a; nb += b * b;
				}
				return dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-12 );
			};
			for ( int rows : { 1, 2, 3, 5, 8, 16 } ) {
				pod::TensorView x1 = pod::make_tensor_2d( xr.ptr<half>(), pod::DType::FP16, rows, H );
				pod::TensorView o1 = pod::make_tensor_2d( outr.ptr<half>(), pod::DType::FP16, rows, H );
				ENGINE_ASSERT( engine::modules::MoE::fused_decode( mw, scratch_r, rows ), "test_moe: the extended experts must take the coop kernels" );
				std::vector<half> got( static_cast<size_t>( rows ) * H ), plain( got.size() );
				engine::modules::MoE::forward( x1, mw, d_idx_r, d_w_r, o1, 0, E, scratch_r, 10.0f, stream );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				CUDA_CHECK( cudaMemcpy( got.data(), o1.data, got.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
				void* const keep = mw.ext_ptrs;
				mw.ext_ptrs = nullptr;
				engine::modules::MoE::forward( x1, mw, d_idx_r, d_w_r, o1, 0, E, scratch_r, 10.0f, stream );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				mw.ext_ptrs = keep;
				CUDA_CHECK( cudaMemcpy( plain.data(), o1.data, plain.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
				const double c = cosine( got, rows ), cp = cosine( plain, rows );
				std::printf( "[*] extended %s order, %2d rows vs reference: cos=%.6f (companions dropped: %.6f)\n", vt ? "Volta" : "stock", rows, c, cp );
				ext_pass = ext_pass && c >= 0.9999 && ( ext_real || cp < 0.9995 );   // a real checkpoint's companions are near the plain decode
			}
			// The persistent kernel against the rows one at a time (the coop kernels, checked above)
			std::vector<half> per_row( static_cast<size_t>( R ) * H );
			for ( int r = 0; r < R; ++r ) {
				pod::TensorView x1 = pod::make_tensor_2d( xr.ptr<half>() + static_cast<size_t>( r ) * H, pod::DType::FP16, 1, H );
				pod::TensorView o1 = pod::make_tensor_2d( outr.ptr<half>(), pod::DType::FP16, 1, H );
				engine::modules::MoE::forward( x1, mw, d_idx_r + r * K, d_w_r + r * K, o1, 0, E, scratch_r, 10.0f, stream );
				CUDA_CHECK( cudaMemcpyAsync( per_row.data() + static_cast<size_t>( r ) * H, o1.data, H * sizeof( half ), cudaMemcpyDeviceToHost, stream ) );
			}
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			auto cos_rows = [&]( const std::vector<half>& got, int rows ) {
				double dot = 0, na = 0, nb = 0;
				for ( size_t i = 0; i < static_cast<size_t>( rows ) * H; ++i ) {
					const double a = __half2float( per_row[i] ), b = __half2float( got[i] );
					dot += a * b; na += a * a; nb += b * b;
				}
				return dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-12 );
			};
			for ( int rows : { 24, 64, 200 } ) {
				pod::TensorView x1 = pod::make_tensor_2d( xr.ptr<half>(), pod::DType::FP16, rows, H );
				pod::TensorView o1 = pod::make_tensor_2d( outr.ptr<half>(), pod::DType::FP16, rows, H );
				std::vector<half> got( static_cast<size_t>( rows ) * H ), plain( got.size() );
				engine::modules::MoE::forward( x1, mw, d_idx_r, d_w_r, o1, 0, E, scratch_r, 10.0f, stream );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				CUDA_CHECK( cudaMemcpy( got.data(), o1.data, got.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
				void* const keep = mw.ext_ptrs;
				mw.ext_ptrs = nullptr;
				engine::modules::MoE::forward( x1, mw, d_idx_r, d_w_r, o1, 0, E, scratch_r, 10.0f, stream );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				mw.ext_ptrs = keep;
				CUDA_CHECK( cudaMemcpy( plain.data(), o1.data, plain.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
				const double c = cos_rows( got, rows ), cp = cos_rows( plain, rows );
				std::printf( "[*] extended %s order, %3d rows (persistent) vs per-row: cos=%.6f (companions dropped: %.6f)\n", vt ? "Volta" : "stock", rows, c, cp );
				ext_pass = ext_pass && c >= 0.9999 && ( ext_real || cp < 0.9995 );   // a real checkpoint's companions are near the plain decode
			}
		}
		std::printf( "[%c] MoE EXTENDED %s\n", ext_pass ? '+' : '-', ext_pass ? "PASS" : "FAIL" );
	}
	repo.close();
	engine::core::arena_destroy( arena );
	engine::modules::trellis_context_destroy();
	(void) cudaDeviceReset();
	return 0;
}
