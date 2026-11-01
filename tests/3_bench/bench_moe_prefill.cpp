// Routed-MoE prefill throughput on real GLM-5.3-Flash layer-3 experts (rank-0 partition, 72 of 288 experts):
// rows routed uniformly over all 288 experts (BENCH_EXPERTS=N: over the first N), time per MoE::forward at several chunk sizes;
// BENCH_RECON=1 also times one expert's three matrices through LinearTrellis::forward_recon (FP16 dequantization + cuBLAS) at those row counts.
//   bench_moe_prefill [rows ...]
#include "engine/runtime/test_paths.h"
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/loader/safetensors.h"
#include "engine/modules/linear.h"
#include "engine/modules/moe.h"
#include "engine/runtime/options.h"
#include "engine/modules/swiglu.h"
#include "trellis_context.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main( int argc, char** argv ) {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	const char* dir = engine::runtime::test::model_path( "ENGINE_TEST_GLM_DIR" );
	const bool shard = std::getenv( "BENCH_SHARD" ) != nullptr; // every expert a quarter wide (rank 0 of 4), all 288 local
	const int H = 4096, I = 2048, E = shard ? 288 : 72, K = 8, L = 3;
	CUDA_CHECK( cudaSetDevice( 0 ) );
	engine::modules::trellis_context_init_all();
	engine::modules::trellis_get_workspace( 0, 16 * 1024 * 1024 );
	engine::modules::trellis_get_locks( 0 );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 2ULL * 1024 * 1024 * 1024 ), "arena" );

	engine::loader::SafeTensorsRepo repo;
	ENGINE_ASSERT( repo.open( dir ), "open model dir" );
	pod::MoELayerWeights mw;
	mw.type = pod::QuantType::TRELLIS;
	mw.num_local_experts = E;
	char name[256];
	for ( int e = 0; e < E; ++e ) {
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.gate_proj", L, e );
		mw.gate[e] = shard ? repo.load_linear_slice( name, arena, 4, 1, 0, 4, stream ) : repo.load_linear( name, arena, 4, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.up_proj", L, e );
		mw.up[e] = shard ? repo.load_linear_slice( name, arena, 4, 1, 0, 4, stream ) : repo.load_linear( name, arena, 4, stream );
		std::snprintf( name, sizeof( name ), "model.language_model.layers.%d.mlp.experts.%d.down_proj", L, e );
		mw.down[e] = shard ? repo.load_linear_slice( name, arena, 4, 0, 0, 4, stream ) : repo.load_linear( name, arena, 4, stream );
	}
	engine::modules::MoE::init_device_ptrs( mw, arena, stream );
	std::printf( "[*] experts: bits g/u/d = %d/%d/%d, mul1 g/u/d = %d/%d/%d\n",
		mw.gate[0].bits, mw.up[0].bits, mw.down[0].bits, mw.gate[0].mul1, mw.up[0].mul1, mw.down[0].mul1 );

	// Reference: the dense decode kernels on one expert matrix (gate: 4096 -> 2048, 3 bit), rows 1..16
	for ( int Rg : { 1, 4, 8, 16 } ) {
		int xs[2] = { Rg, H }, ys[2] = { Rg, I };
		auto xg = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		auto yg = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ys, 2 );
		CUDA_CHECK( cudaMemset( xg.data, 0x3c, static_cast<size_t>( Rg ) * H * 2 ) );
		cudaEvent_t g0, g1;
		CUDA_CHECK( cudaEventCreate( &g0 ) );
		CUDA_CHECK( cudaEventCreate( &g1 ) );
		for ( int it = 0; it < 3; ++it ) engine::modules::Linear::forward( xg, mw.gate[0], yg, stream );
		CUDA_CHECK( cudaEventRecord( g0, stream ) );
		for ( int it = 0; it < 50; ++it ) engine::modules::Linear::forward( xg, mw.gate[it % E], yg, stream );
		CUDA_CHECK( cudaEventRecord( g1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( g1 ) );
		float gms = 0.0f;
		CUDA_CHECK( cudaEventElapsedTime( &gms, g0, g1 ) );
		std::printf( "  dense Linear::forward on an expert gate matrix, %2d rows: %.1f us (%.0f GB/s of 3.2 MB)\n", Rg, gms * 1000.0f / 50.0f, 3.1e6 / ( gms * 1000.0f / 50.0f ) * 1e-3 );
	}
	std::vector<int> row_list;
	for ( int i = 1; i < argc; ++i ) row_list.push_back( std::atoi( argv[i] ) );
	if ( row_list.empty() ) row_list = { 16, 64, 128, 256, 512 };
	uint32_t seed = 12345u;
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	for ( int R : row_list ) {
		int rs[2] = { R, H };
		auto xr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rs, 2 );
		auto outr = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, rs, 2 );
		int* d_idx = static_cast<int*>( engine::core::arena_alloc( arena, R * K * sizeof( int ) ) );
		half* d_w = static_cast<half*>( engine::core::arena_alloc( arena, R * K * sizeof( half ) ) );
		std::vector<half> hx( static_cast<size_t>( R ) * H ), hw( R * K );
		std::vector<int> hi( R * K );
		for ( auto& v : hx ) {
			seed = seed * 1664525u + 1013904223u;
			v = __float2half( ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f );
		}
		int local_routes = 0;
		// BENCH_EXPERTS=N: rows routed over experts 0 .. N-1 only (N >= 8; 8 = every row to the same 8 experts, each `rows` rows: heavy experts)
		static const int route_experts = std::getenv( "BENCH_EXPERTS" ) ? std::max( K, std::atoi( std::getenv( "BENCH_EXPERTS" ) ) ) : 288;
		for ( int r = 0; r < R; ++r ) {
			int used[288] = {};
			for ( int k = 0; k < K; ++k ) {
				int e;
				do {
					seed = seed * 1664525u + 1013904223u;
					e = ( seed >> 8 ) % route_experts;
				} while ( used[e] );
				used[e] = 1;
				hi[r * K + k] = e;
				hw[r * K + k] = __float2half( 0.125f );
				local_routes += e < E ? 1 : 0;
			}
		}
		CUDA_CHECK( cudaMemcpy( xr.data, hx.data(), hx.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_idx, hi.data(), hi.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( d_w, hw.data(), hw.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
		auto scratch = engine::modules::MoE::alloc_scratch( arena, E, K, H, I, R );
		// The 32-row prefill tile (moe_prefill_tile) against the 16-row tile on the same routes: time and agreement
		std::vector<half> res[2];
		for ( int new_tile = 1; new_tile >= 0; --new_tile ) {
			engine::runtime::options().moe_prefill_tile = new_tile != 0;
			for ( int it = 0; it < 2; ++it ) engine::modules::MoE::forward( xr, mw, d_idx, d_w, outr, 0, E, scratch, 10.0f, stream );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			const int iters = 10;
			CUDA_CHECK( cudaEventRecord( e0, stream ) );
			for ( int it = 0; it < iters; ++it ) engine::modules::MoE::forward( xr, mw, d_idx, d_w, outr, 0, E, scratch, 10.0f, stream );
			CUDA_CHECK( cudaEventRecord( e1, stream ) );
			CUDA_CHECK( cudaEventSynchronize( e1 ) );
			float ms = 0.0f;
			CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
			res[new_tile].resize( static_cast<size_t>( R ) * H );
			CUDA_CHECK( cudaMemcpy( res[new_tile].data(), outr.data, res[new_tile].size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			std::printf( "  rows %4d, %s tile: %7.3f ms per layer (%d local routes, %.2f us per local route)\n", R, new_tile ? "32-row" : "16-row", ms / iters, local_routes, 1000.0 * ms / iters / MAX( 1, local_routes ) );
		}
		double dot = 0, na = 0, nb = 0, maxd = 0;
		for ( size_t i = 0; i < res[0].size(); ++i ) {
			const double x = __half2float( res[0][i] ), y = __half2float( res[1][i] );
			dot += x * y; na += x * x; nb += y * y; maxd = std::max( maxd, std::fabs( x - y ) );
		}
		std::printf( "    agreement: cos %.7f, max |diff| %.5f (rms %.4f)\n", dot / std::sqrt( na * nb + 1e-30 ), maxd, std::sqrt( na / res[0].size() ) );
		if ( const int heavy = engine::runtime::options().moe_heavy_rows; heavy > 0 ) {
			// the heavy-expert path (moe_heavy_rows) against the quantized kernel alone, on the same routes
			engine::runtime::options().moe_prefill_tile = true;
			std::vector<half> hv[2];
			for ( int on = 0; on < 2; ++on ) {
				engine::runtime::options().moe_heavy_rows = on ? heavy : 0;
				engine::modules::MoE::forward( xr, mw, d_idx, d_w, outr, 0, E, scratch, 10.0f, stream );
				CUDA_CHECK( cudaStreamSynchronize( stream ) );
				hv[on].resize( static_cast<size_t>( R ) * H );
				CUDA_CHECK( cudaMemcpy( hv[on].data(), outr.data, hv[on].size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			}
			engine::runtime::options().moe_heavy_rows = heavy;
			double d2 = 0, a2 = 0, b2 = 0, md = 0;
			for ( size_t i = 0; i < hv[0].size(); ++i ) {
				const double x = __half2float( hv[0][i] ), y = __half2float( hv[1][i] );
				d2 += x * y; a2 += x * x; b2 += y * y; md = std::max( md, std::fabs( x - y ) );
			}
			std::printf( "    heavy path vs quantized kernel: cos %.7f, max |diff| %.5f (rms %.4f)\n", d2 / std::sqrt( a2 * b2 + 1e-30 ), md, std::sqrt( a2 / hv[0].size() ) );
		}
		if ( std::getenv( "BENCH_RECON" ) ) {
			// one expert's gate, up and down through FP16 dequantization + cuBLAS at R rows (the heavy-expert path)
			int gs[2] = { R, I }, ds[2] = { R, H };
			auto yg2 = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, gs, 2 );
			auto yu2 = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, gs, 2 );
			auto yd2 = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ds, 2 );
			auto rc = [&]( const pod::TensorView& x, const pod::LinearWeight& lw, pod::TensorView& y ) {
				engine::modules::LinearTrellis::forward_recon( x, lw.w0, lw.w1, lw.w2, y, lw.bits, lw.mcg, lw.mul1, stream );
			};
			for ( int it = 0; it < 2; ++it ) { rc( xr, mw.gate[0], yg2 ); rc( xr, mw.up[0], yu2 ); rc( yg2, mw.down[0], yd2 ); }
			CUDA_CHECK( cudaEventRecord( e0, stream ) );
			for ( int it = 0; it < 10; ++it ) { const int e = it % 8; rc( xr, mw.gate[e], yg2 ); rc( xr, mw.up[e], yu2 ); rc( yg2, mw.down[e], yd2 ); }
			CUDA_CHECK( cudaEventRecord( e1, stream ) );
			CUDA_CHECK( cudaEventSynchronize( e1 ) );
			float rms = 0.0f;
			CUDA_CHECK( cudaEventElapsedTime( &rms, e0, e1 ) );
			std::printf( "  rows %4d: one expert through FP16 dequantization + cuBLAS (gate, up, down): %.1f us\n", R, rms * 100.0f );
		}
	}
	return 0;
}
