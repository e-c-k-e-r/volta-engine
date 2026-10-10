// Op-level golden test for the routed MoE module API: MoERouter::route_topk (router logits, sigmoid, biased top-k) and
// MoE::forward (per-expert gate/up -> SwiGLU -> down, weighted sum), full expert range and a sharded range [4, 12).
// ONE source for both configurations:
//   --dump <dir>   run in the CUDA build: computes with the real kernels, writes golden outputs
//   --check <dir>  recompute with the built backend and compare (tier 2) against the golden files
// 3-bit MUL1 trellis experts (what the persistent / fused CUDA kernels instantiate), rows 1, 4, 17.
// --softmax routes with MoERouter::route_softmax_topk (the Qwen family's router) instead of route_topk (GLM's sigmoid router); the case names then carry
// an _sm suffix, so one golden directory serves both.
#include "golden_io.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/backend/tiers.h"
#include "engine/core/arena.h"
#include "engine/modules/moe.h"
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

static pod::MoELayerWeights g_full, g_shard;

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
	modules::MoE::init_device_ptrs( g_full, g_arena );
	modules::MoE::init_device_ptrs( g_shard, g_arena );
	const pod::MoEScratch sc_full = modules::MoE::alloc_scratch( g_arena, E, TOPK, H, I, max_rows );
	const pod::MoEScratch sc_shard = modules::MoE::alloc_scratch( g_arena, 8, TOPK, H, I, max_rows );

	auto router_w = upload_half( uniform_vec( rng, static_cast<size_t>( H ) * E, 0.2f ), { H, E } );
	auto router_bias = upload_half( uniform_vec( rng, E, 0.1f ), { E } );

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
			pass = golden_compare( c.name.c_str(), ref, c.out ) && pass;
		}
	}
	core::arena_destroy( g_arena );
	std::printf( "%s\n", pass ? "ALL PASS" : "FAILED" );
	return pass ? 0 : 1;
}
