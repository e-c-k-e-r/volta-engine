// Op-level golden test for the routed MoE module API: MoERouter::route_topk (router logits, sigmoid, biased top-k) and
// MoE::forward (per-expert gate/up -> SwiGLU -> down, weighted sum), full expert range and a sharded range [4, 12).
// ONE source for both configurations:
//   --dump <dir>   run in the CUDA build: computes with the real kernels, writes golden outputs
//   --check <dir>  recompute with the built backend and compare (tier 2) against the golden files
// 3-bit MUL1 trellis experts (what the persistent / fused CUDA kernels instantiate), the same with the extended trellis' tile starts and gains (both,
// starts only or gains only by expert, in the stock and the Volta tile order) and GGUF experts (Q4_K gate, IQ3_S up, Q6_K down), rows 1, 4, 17.
// --softmax routes with MoERouter::route_softmax_topk (the Qwen family's router) instead of route_topk (GLM's sigmoid router); the case names then carry
// an _sm suffix, so one golden directory serves both.
#include "golden_io.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/backend/tiers.h"
#include "engine/core/arena.h"
#include "engine/modules/moe.h"
#include "engine/modules/gguf.h"
#include <algorithm>
#include <cstring>
#include <vector>

using namespace engine;
using backend::half_t;

static pod::MemoryArena g_arena{};

static pod::TensorView upload_half( const std::vector<float>& v, std::initializer_list<int> shape ) {
	int s[4] = {};
	int nd = 0;
	for ( int d : shape ) s[nd++] = d;
	pod::TensorView t = core::arena_alloc_tensor( g_arena, pod::DType::FP16, s, nd );
	std::vector<half_t> h( v.size() );
	for ( size_t i = 0; i < v.size(); ++i ) h[i] = backend::from_float_half( v[i] );
	backend::memcpy_sync( t.data, h.data(), h.size() * sizeof( half_t ), backend::Copy::H2D );
	return t;
}

// FP32 values as the FP16 upload would hold them (the router's selection bias is FP32; the goldens were taken with FP16-rounded values)
static pod::TensorView upload_half_values( const std::vector<float>& v, std::initializer_list<int> shape ) {
	int s[4] = {};
	int nd = 0;
	for ( int d : shape ) s[nd++] = d;
	pod::TensorView t = core::arena_alloc_tensor( g_arena, pod::DType::FP32, s, nd );
	std::vector<float> f( v.size() );
	for ( size_t i = 0; i < v.size(); ++i ) f[i] = backend::to_float( backend::from_float_half( v[i] ) );
	backend::memcpy_sync( t.data, f.data(), f.size() * sizeof( float ), backend::Copy::H2D );
	return t;
}

static pod::TensorView upload_codes( GoldenRng& rng, int k, int n, int bits ) {
	std::vector<uint16_t> c( static_cast<size_t>( k / 16 ) * ( n / 16 ) * 16 * bits );
	for ( auto& v : c ) v = static_cast<uint16_t>( rng.next() );
	int s[3] = { k / 16, n / 16, 16 * bits };
	pod::TensorView t = core::arena_alloc_tensor( g_arena, pod::DType::UINT16, s, 3 );
	backend::memcpy_sync( t.data, c.data(), c.size() * sizeof( uint16_t ), backend::Copy::H2D );
	return t;
}

static std::vector<float> uniform_vec( GoldenRng& rng, size_t n, float scale ) {
	std::vector<float> v( n );
	for ( auto& x : v ) x = rng.uniform( scale );
	return v;
}

static std::vector<float> sign_vec( GoldenRng& rng, size_t n ) {
	std::vector<float> v( n );
	for ( auto& x : v ) x = ( rng.next() >> 4 ) & 1 ? 1.0f : -1.0f;
	return v;
}

static std::vector<float> download_half( const void* dev, size_t n ) {
	std::vector<half_t> h( n );
	backend::device_sync();
	backend::memcpy_sync( h.data(), dev, n * sizeof( half_t ), backend::Copy::D2H );
	std::vector<float> f( n );
	for ( size_t i = 0; i < n; ++i ) f[i] = backend::to_float( h[i] );
	return f;
}

static pod::LinearWeight make_expert_weight( GoldenRng& rng, int k, int n, int bits ) {
	auto codes = upload_codes( rng, k, n, bits );
	auto suh = upload_half( sign_vec( rng, k ), { k } );
	auto svh = upload_half( sign_vec( rng, n ), { n } );
	return pod::LinearWeight::make_trellis( codes, suh, svh, bits, true, false );
}

// An extended trellis expert: random tile start states ([k / 16, n / 16] uint16) and gain codes (uint8) where asked
static pod::LinearWeight make_ext_expert_weight( GoldenRng& rng, int k, int n, int bits, bool starts, bool gains, bool volta ) {
	pod::LinearWeight lw = make_expert_weight( rng, k, n, bits );
	lw.volta_tiles = volta;
	int s[2] = { k / 16, n / 16 };
	const size_t tiles = static_cast<size_t>( k / 16 ) * ( n / 16 );
	if ( starts ) {
		std::vector<uint16_t> st( tiles );
		for ( auto& v : st ) v = static_cast<uint16_t>( rng.next() );
		lw.w3 = core::arena_alloc_tensor( g_arena, pod::DType::UINT16, s, 2 );
		backend::memcpy_sync( lw.w3.data, st.data(), st.size() * sizeof( uint16_t ), backend::Copy::H2D );
	}
	if ( gains ) {
		std::vector<uint8_t> gn( tiles );
		for ( auto& v : gn ) v = static_cast<uint8_t>( rng.next() );
		lw.w4 = core::arena_alloc_tensor( g_arena, pod::DType::UINT8, s, 2 );
		backend::memcpy_sync( lw.w4.data, gn.data(), gn.size(), backend::Copy::H2D );
	}
	return lw;
}

// A GGUF expert: n rows of k / 256 blocks of random bytes (valid codes for these types) with the half scales set small, so the products stay
// in FP16 range; d_at / dmin_at are the byte offsets of the scales in a block (dmin_at < 0: none)
static pod::LinearWeight make_gguf_expert( GoldenRng& rng, int k, int n, modules::GgmlType type, int block_bytes, int d_at, int dmin_at ) {
	std::vector<uint8_t> b( static_cast<size_t>( n ) * ( k / 256 ) * block_bytes );
	for ( auto& v : b ) v = static_cast<uint8_t>( rng.next() );
	const half_t d = backend::from_float_half( 1e-4f ), dmin = backend::from_float_half( 5e-5f );
	for ( size_t o = 0; o < b.size(); o += block_bytes ) {
		std::memcpy( &b[o + d_at], &d, 2 );
		if ( dmin_at >= 0 ) std::memcpy( &b[o + dmin_at], &dmin, 2 );
	}
	int s[1] = { static_cast<int>( b.size() ) };
	pod::TensorView t = core::arena_alloc_tensor( g_arena, pod::DType::UINT8, s, 1 );
	backend::memcpy_sync( t.data, b.data(), b.size(), backend::Copy::H2D );
	return pod::LinearWeight::make_gguf( t, static_cast<uint32_t>( type ), k, n );
}

// An NVFP4 expert in the loader's prepacked layout (codes [n, k / 2] bytes, block scales [n, k / 16]): random code bytes, E4M3 block scales with
// exponents 5 - 9 (no NaN), a small global scale so the products stay in FP16 range
static pod::LinearWeight make_nvfp4_expert( GoldenRng& rng, int k, int n, float alpha ) {
	std::vector<uint8_t> hc( static_cast<size_t>( n ) * k / 2 ), hs( static_cast<size_t>( n ) * k / 16 );
	for ( auto& v : hc ) v = static_cast<uint8_t>( rng.next() );
	for ( auto& v : hs ) v = static_cast<uint8_t>( ( ( rng.next() & 1 ) << 7 ) | ( ( 5 + rng.next() % 5 ) << 3 ) | ( rng.next() & 7 ) );
	int cs[2] = { n, k / 2 }, ss[2] = { n, k / 16 };
	pod::TensorView codes = core::arena_alloc_tensor( g_arena, pod::DType::UINT8, cs, 2 );
	pod::TensorView scales = core::arena_alloc_tensor( g_arena, pod::DType::UINT8, ss, 2 );
	backend::memcpy_sync( codes.data, hc.data(), hc.size(), backend::Copy::H2D );
	backend::memcpy_sync( scales.data, hs.data(), hs.size(), backend::Copy::H2D );
	return pod::LinearWeight::make_nvfp4( codes, scales, alpha );
}

static pod::MoELayerWeights g_full, g_shard, g_gguf_full, g_gguf_shard, g_nv_full, g_nv_shard, g_ext_full[2], g_ext_shard[2];

struct Case {
	std::string name;
	std::vector<float> out;
};

int main( int argc, char** argv ) {
	GoldenArgs args( argc, argv );
	bool softmax = false;
	for ( int i = 1; i < argc; ++i ) if ( !std::strcmp( argv[i], "--softmax" ) ) softmax = true;
	if ( !args.dump && !args.check ) args.dump = args.check = true;
	ENGINE_ASSERT( core::arena_create( g_arena, 0, 256ull << 20 ), "arena" );
	std::printf( "=== op-level routed MoE (backend %s) ===\n", ENGINE_BACKEND_NAME );

	const int H = 1024, I = 512, E = 16, TOPK = 8;
	const int row_list[] = { 1, 4, 17 };
	const int max_rows = 17;
	GoldenRng rng( 4242u );

	g_full.type = pod::QuantType::TRELLIS;
	g_full.num_local_experts = E;
	for ( int e = 0; e < E; ++e ) {
		g_full.gate[e] = make_expert_weight( rng, H, I, 3 );
		g_full.up[e] = make_expert_weight( rng, H, I, 3 );
		g_full.down[e] = make_expert_weight( rng, I, H, 3 );
	}
	// Sharded range: experts [4, 12) as a table of 8 local experts (local index = expert - start)
	g_shard.type = pod::QuantType::TRELLIS;
	g_shard.num_local_experts = 8;
	for ( int e = 0; e < 8; ++e ) {
		g_shard.gate[e] = g_full.gate[4 + e];
		g_shard.up[e] = g_full.up[4 + e];
		g_shard.down[e] = g_full.down[4 + e];
	}
	// GGUF experts (MoE::forward with QuantType::GGUF): gate Q4_K, up IQ3_S, down Q6_K
	g_gguf_full.type = g_gguf_shard.type = pod::QuantType::GGUF;
	g_gguf_full.num_local_experts = E;
	g_gguf_shard.num_local_experts = 8;
	g_gguf_full.top_k = g_gguf_shard.top_k = TOPK;
	// part by part, so each part's experts are one stack (Vulkan runs up to 8 rows on its expert-indexed shaders, more on the host path)
	for ( int e = 0; e < E; ++e ) g_gguf_full.gate[e] = make_gguf_expert( rng, H, I, modules::GgmlType::Q4_K, 144, 0, 2 );
	for ( int e = 0; e < E; ++e ) g_gguf_full.up[e] = make_gguf_expert( rng, H, I, modules::GgmlType::IQ3_S, 110, 0, -1 );
	for ( int e = 0; e < E; ++e ) g_gguf_full.down[e] = make_gguf_expert( rng, I, H, modules::GgmlType::Q6_K, 210, 208, -1 );
	for ( int e = 0; e < 8; ++e ) {
		g_gguf_shard.gate[e] = g_gguf_full.gate[4 + e];
		g_gguf_shard.up[e] = g_gguf_full.up[4 + e];
		g_gguf_shard.down[e] = g_gguf_full.down[4 + e];
	}
	modules::MoE::init_device_ptrs( g_gguf_full, g_arena );
	modules::MoE::init_device_ptrs( g_gguf_shard, g_arena );
	pod::MoEScratch sc_gguf{};
	sc_gguf.gguf_ws = core::arena_alloc( g_arena, std::max<size_t>( 256, modules::MoEGguf::scratch_bytes( TOPK, H, I ) ), 256 );
	modules::MoE::init_device_ptrs( g_full, g_arena );
	modules::MoE::init_device_ptrs( g_shard, g_arena );
	const pod::MoEScratch sc_full = modules::MoE::alloc_scratch( g_arena, E, TOPK, H, I, max_rows );
	const pod::MoEScratch sc_shard = modules::MoE::alloc_scratch( g_arena, 8, TOPK, H, I, max_rows );
	// NVFP4 experts (CUDA: the NVFP4 expert kernels; Vulkan: its device kernels up to 16 rows, the host path beyond; CPU: the host path)
	g_nv_full.type = g_nv_shard.type = pod::QuantType::NVFP4;
	g_nv_full.num_local_experts = E;
	g_nv_shard.num_local_experts = 8;
	g_nv_full.top_k = g_nv_shard.top_k = TOPK;
	for ( int e = 0; e < E; ++e ) {
		g_nv_full.gate[e] = make_nvfp4_expert( rng, H, I, 0.002f );
		g_nv_full.up[e] = make_nvfp4_expert( rng, H, I, 0.0025f );
		g_nv_full.down[e] = make_nvfp4_expert( rng, I, H, 0.003f );
	}
	for ( int e = 0; e < 8; ++e ) {
		g_nv_shard.gate[e] = g_nv_full.gate[4 + e];
		g_nv_shard.up[e] = g_nv_full.up[4 + e];
		g_nv_shard.down[e] = g_nv_full.down[4 + e];
	}
	modules::MoE::init_device_ptrs( g_nv_full, g_arena );
	modules::MoE::init_device_ptrs( g_nv_shard, g_arena );
	const pod::MoEScratch sc_nv_full = modules::MoE::alloc_scratch( g_arena, E, TOPK, H, I, max_rows, true );
	const pod::MoEScratch sc_nv_shard = modules::MoE::alloc_scratch( g_arena, 8, TOPK, H, I, max_rows, true );
	// Extended trellis experts: expert e has both companions (e % 3 == 0), starts only (1) or gains only (2); [0] stock, [1] Volta tile order
	for ( int vt = 0; vt < 2; ++vt ) {
		g_ext_full[vt].type = g_ext_shard[vt].type = pod::QuantType::TRELLIS;
		g_ext_full[vt].num_local_experts = E;
		g_ext_shard[vt].num_local_experts = 8;
		g_ext_full[vt].top_k = g_ext_shard[vt].top_k = TOPK;
		for ( int e = 0; e < E; ++e ) {
			const bool st = e % 3 != 2, gn = e % 3 != 1;
			g_ext_full[vt].gate[e] = make_ext_expert_weight( rng, H, I, 3, st, gn, vt == 1 );
			g_ext_full[vt].up[e] = make_ext_expert_weight( rng, H, I, 3, st, gn, vt == 1 );
			g_ext_full[vt].down[e] = make_ext_expert_weight( rng, I, H, 3, st, gn, vt == 1 );
		}
		for ( int e = 0; e < 8; ++e ) {
			g_ext_shard[vt].gate[e] = g_ext_full[vt].gate[4 + e];
			g_ext_shard[vt].up[e] = g_ext_full[vt].up[4 + e];
			g_ext_shard[vt].down[e] = g_ext_full[vt].down[4 + e];
		}
		modules::MoE::init_device_ptrs( g_ext_full[vt], g_arena );
		modules::MoE::init_device_ptrs( g_ext_shard[vt], g_arena );
	}

	auto router_w = upload_half( uniform_vec( rng, static_cast<size_t>( H ) * E, 0.2f ), { H, E } );
	auto router_bias = upload_half_values( uniform_vec( rng, E, 0.1f ), { E } );

	std::vector<Case> cases;
	for ( int m : row_list ) {
		auto x = upload_half( uniform_vec( rng, static_cast<size_t>( m ) * H, 1.0f ), { m, H } );
		int* d_idx = static_cast<int*>( core::arena_alloc( g_arena, static_cast<size_t>( m ) * TOPK * sizeof( int ) ) );
		half_t* d_w = static_cast<half_t*>( core::arena_alloc( g_arena, static_cast<size_t>( m ) * TOPK * sizeof( half_t ) ) );
		if ( softmax ) modules::MoERouter::route_softmax_topk( x, router_w, d_idx, d_w, H, E, TOPK, nullptr, m );
		else modules::MoERouter::route_topk( x, router_w, router_bias, d_idx, d_w, H, E, TOPK, 2.5f, nullptr, m );
		backend::device_sync();
		std::vector<int> idx( static_cast<size_t>( m ) * TOPK );
		backend::memcpy_sync( idx.data(), d_idx, idx.size() * sizeof( int ), backend::Copy::D2H );
		std::vector<float> idxf( idx.begin(), idx.end() );
		char nm[64];
		std::snprintf( nm, sizeof( nm ), "route_idx%s_m%d", softmax ? "_sm" : "", m );
		cases.push_back( { nm, idxf } );
		std::snprintf( nm, sizeof( nm ), "route_w%s_m%d", softmax ? "_sm" : "", m );
		cases.push_back( { nm, download_half( d_w, idx.size() ) } );

		for ( int shard = 0; shard < 2; ++shard ) {
			int os[2] = { m, H };
			pod::TensorView out = core::arena_alloc_tensor( g_arena, pod::DType::FP16, os, 2 );
			if ( shard ) modules::MoE::forward( x, g_shard, d_idx, d_w, out, 4, 12, sc_shard );
			else modules::MoE::forward( x, g_full, d_idx, d_w, out, 0, E, sc_full );
			std::snprintf( nm, sizeof( nm ), "moe%s_%s_m%d", softmax ? "_sm" : "", shard ? "shard" : "full", m );
			cases.push_back( { nm, download_half( out.data, out.numel() ) } );
		}
		for ( int shard = 0; shard < 2; ++shard ) {
			int os[2] = { m, H };
			pod::TensorView out = core::arena_alloc_tensor( g_arena, pod::DType::FP16, os, 2 );
			if ( shard ) modules::MoE::forward( x, g_gguf_shard, d_idx, d_w, out, 4, 12, sc_gguf );
			else modules::MoE::forward( x, g_gguf_full, d_idx, d_w, out, 0, E, sc_gguf );
			std::snprintf( nm, sizeof( nm ), "gguf%s_%s_m%d", softmax ? "_sm" : "", shard ? "shard" : "full", m );
			cases.push_back( { nm, download_half( out.data, out.numel() ) } );
		}
		for ( int shard = 0; shard < 2; ++shard ) {
			int os[2] = { m, H };
			pod::TensorView out = core::arena_alloc_tensor( g_arena, pod::DType::FP16, os, 2 );
			if ( shard ) modules::MoE::forward( x, g_nv_shard, d_idx, d_w, out, 4, 12, sc_nv_shard );
			else modules::MoE::forward( x, g_nv_full, d_idx, d_w, out, 0, E, sc_nv_full );
			std::snprintf( nm, sizeof( nm ), "nvfp4%s_%s_m%d", softmax ? "_sm" : "", shard ? "shard" : "full", m );
			cases.push_back( { nm, download_half( out.data, out.numel() ) } );
		}
#ifndef USE_ROCM   // the extended trellis (tile starts / gains) runs on the CUDA coop kernels only (docs/ROCM.md): no ROCm golden for it
		for ( int vt = 0; vt < 2; ++vt ) {
			for ( int shard = 0; shard < 2; ++shard ) {
				int os[2] = { m, H };
				pod::TensorView out = core::arena_alloc_tensor( g_arena, pod::DType::FP16, os, 2 );
				if ( shard ) modules::MoE::forward( x, g_ext_shard[vt], d_idx, d_w, out, 4, 12, sc_shard );
				else modules::MoE::forward( x, g_ext_full[vt], d_idx, d_w, out, 0, E, sc_full );
				std::snprintf( nm, sizeof( nm ), "ext%s%s_%s_m%d", vt ? "_volta" : "", softmax ? "_sm" : "", shard ? "shard" : "full", m );
				cases.push_back( { nm, download_half( out.data, out.numel() ) } );
			}
		}
#endif
	}

	{ // accumulate (accum += w * expert_out) and combine (shared + sum_k w[k] * part[k])
		auto accum = upload_half( uniform_vec( rng, H, 1.0f ), { H } );
		auto eo = upload_half( uniform_vec( rng, H, 1.0f ), { H } );
		modules::MoERouter::accumulate( accum.ptr<half_t>(), eo.ptr<const half_t>(), backend::from_float_half( 0.375f ), H, nullptr );
		cases.push_back( { "accumulate", download_half( accum.data, H ) } );
		auto parts = upload_half( uniform_vec( rng, static_cast<size_t>( TOPK ) * H, 1.0f ), { TOPK, H } );
		auto wts = upload_half( uniform_vec( rng, TOPK, 0.5f ), { TOPK } );
		auto shared = upload_half( uniform_vec( rng, H, 1.0f ), { H } );
		auto comb = upload_half( std::vector<float>( H, 0.0f ), { H } );
		modules::MoERouter::combine( parts.ptr<const half_t>(), wts.ptr<const half_t>(), shared.ptr<const half_t>(), comb.ptr<half_t>(), H, TOPK, nullptr );
		cases.push_back( { "combine", download_half( comb.data, H ) } );
	}

	bool pass = true;
	if ( args.dump ) {
		for ( const auto& c : cases ) pass = golden_write( args.dir, "moe_" + c.name, c.out.data(), c.out.size() * sizeof( float ) ) && pass;
		std::printf( "  dumped %zu cases to %s\n", cases.size(), args.dir.c_str() );
	}
	if ( args.check ) {
		for ( const auto& c : cases ) {
			std::vector<uint8_t> raw;
			if ( !golden_read( args.dir, "moe_" + c.name, raw ) ) {
				std::printf( "  %-28s golden missing  FAIL\n", c.name.c_str() );
				pass = false;
				continue;
			}
			std::vector<float> ref( raw.size() / sizeof( float ) );
			std::memcpy( ref.data(), raw.data(), ref.size() * sizeof( float ) );
			// GGUF experts: each backend quantizes the activations its own way (CUDA q8_1, CPU q8_K, Vulkan FP16 / q8 by row count)
			const bool gguf = c.name.rfind( "gguf", 0 ) == 0;
			pass = ( gguf ? golden_compare( c.name.c_str(), ref, c.out, 0.999, 0.05 ) : golden_compare( c.name.c_str(), ref, c.out ) ) && pass;
		}
	}
	core::arena_destroy( g_arena );
	std::printf( "%s\n", pass ? "ALL PASS" : "FAILED" );
	return pass ? 0 : 1;
}
