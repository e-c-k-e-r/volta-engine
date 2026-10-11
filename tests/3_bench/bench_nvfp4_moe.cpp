// Fused NVFP4 routed-expert path (gate/up/SiLU + down + weighted sum, one row, 2 launches) on Flash-Next's expert shapes with synthetic weights:
// parity against the chained LinearNVFP4 calls, then time per layer (graph replay over 32 distinct route sets) as the number of local routes varies.
// Compare with bench_moe_routes (EXL3 trellis experts: ~141 us per layer at 10 local routes).
//   bench_nvfp4_moe
#include "engine/config.h"
#include "engine/core/tensor.h"
#include "engine/core/arena.h"
#include "engine/modules/nvfp4.h"
#include "engine/modules/nvfp4_moe.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint32_t g_seed = 12345u;
static uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

int main() {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 2ULL * 1024 * 1024 * 1024 ), "arena" );
	const int H = 2560, I = 640, E = 128, K = 10, SETS = 32;
	auto alloc_matrix = [&]( int k, int n, pod::TensorView& c, pod::TensorView& s ) {
		int cs[2] = { n, k / 2 }, ss[2] = { n, k / 16 };
		c = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, cs, 2 );
		s = engine::core::arena_alloc_tensor( arena, pod::DType::UINT8, ss, 2 );
		std::vector<uint8_t> hc( static_cast<size_t>( n ) * k / 2 ), hs( static_cast<size_t>( n ) * k / 16 );
		for ( auto& v : hc ) v = static_cast<uint8_t>( rnd() );
		for ( auto& v : hs ) v = static_cast<uint8_t>( 0x30 + ( rnd() & 7 ) );   // E4M3 scales around 0.5..1.5
		CUDA_CHECK( cudaMemcpy( c.data, hc.data(), hc.size(), cudaMemcpyHostToDevice ) );
		CUDA_CHECK( cudaMemcpy( s.data, hs.data(), hs.size(), cudaMemcpyHostToDevice ) );
	};
	std::vector<pod::TensorView> gc( E ), gs( E ), uc( E ), us( E ), dc( E ), ds( E );
	std::vector<const uint8_t*> pgc( E ), pgs( E ), puc( E ), pus( E ), pdc( E ), pds( E );
	std::vector<float> gg( E ), ug( E ), dg( E );
	for ( int e = 0; e < E; ++e ) {
		alloc_matrix( H, I, gc[e], gs[e] ); alloc_matrix( H, I, uc[e], us[e] ); alloc_matrix( I, H, dc[e], ds[e] );
		pgc[e] = gc[e].ptr<const uint8_t>(); pgs[e] = gs[e].ptr<const uint8_t>(); puc[e] = uc[e].ptr<const uint8_t>(); pus[e] = us[e].ptr<const uint8_t>();
		pdc[e] = dc[e].ptr<const uint8_t>(); pds[e] = ds[e].ptr<const uint8_t>();
		gg[e] = 0.004f + 0.0001f * ( e % 5 ); ug[e] = 0.005f; dg[e] = 0.006f;
	}
	auto upload = [&]( const void* p, size_t bytes ) { void* d = engine::core::arena_alloc( arena, bytes ); CUDA_CHECK( cudaMemcpy( d, p, bytes, cudaMemcpyHostToDevice ) ); return d; };
	engine::modules::NVFP4ExpertTable t;
	t.gate_c = static_cast<const uint8_t* const*>( upload( pgc.data(), E * sizeof( void* ) ) );
	t.gate_s = static_cast<const uint8_t* const*>( upload( pgs.data(), E * sizeof( void* ) ) );
	t.up_c = static_cast<const uint8_t* const*>( upload( puc.data(), E * sizeof( void* ) ) );
	t.up_s = static_cast<const uint8_t* const*>( upload( pus.data(), E * sizeof( void* ) ) );
	t.down_c = static_cast<const uint8_t* const*>( upload( pdc.data(), E * sizeof( void* ) ) );
	t.down_s = static_cast<const uint8_t* const*>( upload( pds.data(), E * sizeof( void* ) ) );
	t.gate_g = static_cast<const float*>( upload( gg.data(), E * sizeof( float ) ) );
	t.up_g = static_cast<const float*>( upload( ug.data(), E * sizeof( float ) ) );
	t.down_g = static_cast<const float*>( upload( dg.data(), E * sizeof( float ) ) );
	t.num_local = E; t.min_expert = 0; t.hidden = H; t.inter = I;

	std::vector<half> hx( H );
	for ( auto& v : hx ) v = __float2half( ( rnd() / 16777216.0f ) * 2.0f - 1.0f );
	half* x = static_cast<half*>( upload( hx.data(), H * sizeof( half ) ) );
	std::vector<half> hw( K );
	for ( auto& v : hw ) v = __float2half( 0.05f + 0.01f * ( rnd() % 10 ) );
	half* w = static_cast<half*>( upload( hw.data(), K * sizeof( half ) ) );
	int* idx = static_cast<int*>( engine::core::arena_alloc( arena, SETS * K * sizeof( int ) ) );
	half* hbuf = static_cast<half*>( engine::core::arena_alloc( arena, K * I * sizeof( half ) ) );
	float* partial = static_cast<float*>( engine::core::arena_alloc( arena, K * H * sizeof( float ) ) );
	int* disp = static_cast<int*>( engine::core::arena_alloc( arena, engine::modules::nvfp4_disp_ints( E ) * sizeof( int ) ) );
	int* counters = static_cast<int*>( engine::core::arena_alloc( arena, H / 32 * sizeof( int ) ) );
	half* out = static_cast<half*>( engine::core::arena_alloc( arena, H * sizeof( half ) ) );
	CUDA_CHECK( cudaMemset( counters, 0, H / 32 * sizeof( int ) ) );

	auto make_routes = [&]( int n ) {
		std::vector<int> hi( SETS * K );
		for ( int s = 0; s < SETS; ++s ) {
			bool used[E] = {};
			for ( int k = 0; k < K; ++k ) {
				int e;
				if ( k < n ) { do { e = rnd() % E; } while ( used[e] ); used[e] = true; }
				else e = E;   // another rank's expert
				hi[s * K + k] = e;
			}
		}
		CUDA_CHECK( cudaMemcpy( idx, hi.data(), hi.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
		return hi;
	};

	// Parity (10 local routes, first set) against LinearNVFP4 chains
	{
		auto hi = make_routes( K );
		engine::modules::NVFP4MoE::forward_rows( t, x, idx, w, 1, K, hbuf, partial, disp, counters, nullptr, 0.0f, out, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		std::vector<half> got( H );
		CUDA_CHECK( cudaMemcpy( got.data(), out, H * sizeof( half ), cudaMemcpyDeviceToHost ) );
		int xs[2] = { 1, H }, is[2] = { 1, I };
		pod::TensorView xv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		pod::TensorView gv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, is, 2 );
		pod::TensorView uv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, is, 2 );
		pod::TensorView hv = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, is, 2 );
		pod::TensorView ov = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
		CUDA_CHECK( cudaMemcpy( xv.data, hx.data(), H * sizeof( half ), cudaMemcpyHostToDevice ) );
		std::vector<float> ref( H, 0.f );
		std::vector<half> hg( I ), hu( I ), hh( I ), ho( H );
		for ( int k = 0; k < K; ++k ) {
			const int e = hi[k];
			engine::modules::LinearNVFP4::forward( xv, gc[e], gs[e], gg[e], gv, stream );
			engine::modules::LinearNVFP4::forward( xv, uc[e], us[e], ug[e], uv, stream );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			CUDA_CHECK( cudaMemcpy( hg.data(), gv.data, I * sizeof( half ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( hu.data(), uv.data, I * sizeof( half ), cudaMemcpyDeviceToHost ) );
			for ( int i = 0; i < I; ++i ) { const float g = __half2float( hg[i] ); hh[i] = __float2half( g / ( 1.f + std::exp( -g ) ) * __half2float( hu[i] ) ); }
			CUDA_CHECK( cudaMemcpy( hv.data, hh.data(), I * sizeof( half ), cudaMemcpyHostToDevice ) );
			engine::modules::LinearNVFP4::forward( hv, dc[e], ds[e], dg[e], ov, stream );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			CUDA_CHECK( cudaMemcpy( ho.data(), ov.data, H * sizeof( half ), cudaMemcpyDeviceToHost ) );
			for ( int n = 0; n < H; ++n ) ref[n] += __half2float( hw[k] ) * __half2float( ho[n] );
		}
		double dot = 0, na = 0, nb = 0, maxabs = 0;
		for ( int n = 0; n < H; ++n ) { const double a = ref[n], b = __half2float( got[n] ); dot += a * b; na += a * a; nb += b * b; maxabs = std::fmax( maxabs, std::fabs( a - b ) ); }
		std::printf( "[parity] cosine %.6f  max |diff| %.4g  (ref rms %.4g)\n", dot / std::sqrt( na * nb ), maxabs, std::sqrt( na / H ) );
	}

	// Multi-row: routes drawn from 512 experts (rank 0 owns 0..127, the rest are skipped); rows > 1 against single-row calls, then time per layer
	for ( int R : { 2, 4, 8, 16, 64 } ) {
		std::vector<half> hxr( static_cast<size_t>( R ) * H ), hwr( static_cast<size_t>( R ) * K );
		for ( auto& v : hxr ) v = __float2half( ( rnd() / 16777216.0f ) * 2.0f - 1.0f );
		for ( auto& v : hwr ) v = __float2half( 0.05f + 0.01f * ( rnd() % 10 ) );
		std::vector<int> hir( static_cast<size_t>( R ) * K );
		int local = 0;
		for ( int r = 0; r < R; ++r ) {
			bool used[512] = {};
			for ( int k = 0; k < K; ++k ) {
				int e;
				do { e = rnd() % 512; } while ( used[e] );
				used[e] = true;
				hir[r * K + k] = e;
				local += e < E ? 1 : 0;
			}
		}
		half* xr = static_cast<half*>( upload( hxr.data(), hxr.size() * sizeof( half ) ) );
		half* wr = static_cast<half*>( upload( hwr.data(), hwr.size() * sizeof( half ) ) );
		int* ir = static_cast<int*>( upload( hir.data(), hir.size() * sizeof( int ) ) );
		half* outr = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * H * sizeof( half ) ) );
		half* hb = static_cast<half*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * K * I * sizeof( half ) ) );
		float* pb = static_cast<float*>( engine::core::arena_alloc( arena, static_cast<size_t>( R ) * K * H * sizeof( float ) ) );
		engine::modules::NVFP4MoE::forward_rows( t, xr, ir, wr, R, K, hb, pb, disp, counters, nullptr, 0.0f, outr, stream );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		std::vector<half> got( static_cast<size_t>( R ) * H ), one( H );
		CUDA_CHECK( cudaMemcpy( got.data(), outr, got.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
		size_t bad = 0;
		for ( int r = 0; r < R; ++r ) {
			engine::modules::NVFP4MoE::forward_rows( t, xr + static_cast<size_t>( r ) * H, ir + r * K, wr + r * K, 1, K, hbuf, partial, disp, counters, nullptr, 0.0f, out, stream );
			CUDA_CHECK( cudaStreamSynchronize( stream ) );
			CUDA_CHECK( cudaMemcpy( one.data(), out, H * sizeof( half ), cudaMemcpyDeviceToHost ) );
			for ( int n = 0; n < H; ++n ) bad += __half_as_ushort( one[n] ) != __half_as_ushort( got[static_cast<size_t>( r ) * H + n] );
		}
		cudaGraph_t g;
		cudaGraphExec_t ex;
		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		for ( int s = 0; s < 16; ++s ) engine::modules::NVFP4MoE::forward_rows( t, xr, ir, wr, R, K, hb, pb, disp, counters, nullptr, 0.0f, outr, stream );
		CUDA_CHECK( cudaStreamEndCapture( stream, &g ) );
		CUDA_CHECK( cudaGraphInstantiate( &ex, g, nullptr, nullptr, 0 ) );
		CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		cudaEvent_t a0, a1;
		CUDA_CHECK( cudaEventCreate( &a0 ) );
		CUDA_CHECK( cudaEventCreate( &a1 ) );
		CUDA_CHECK( cudaEventRecord( a0, stream ) );
		for ( int i = 0; i < 20; ++i ) CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
		CUDA_CHECK( cudaEventRecord( a1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( a1 ) );
		float ms = 0.f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, a0, a1 ) );
		std::printf( "  rows %2d: %d local routes, %7.1f us per layer, mismatching halves vs single-row calls: %zu\n", R, local, 1000.0 * ms / 20 / 16, bad );
		CUDA_CHECK( cudaGraphExecDestroy( ex ) );
		CUDA_CHECK( cudaGraphDestroy( g ) );
	}

	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	const double expert_bytes = 3.0 * H * I * ( 0.5 + 1.0 / 16.0 );
	std::printf( "[*] one expert: %.2f MB (codes + scales)\n", expert_bytes / 1e6 );
	for ( int n : { 1, 2, 4, 6, 8, 10 } ) {
		make_routes( n );
		cudaGraph_t g;
		cudaGraphExec_t ex;
		CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
		for ( int s = 0; s < SETS; ++s ) engine::modules::NVFP4MoE::forward_rows( t, x, idx + s * K, w, 1, K, hbuf, partial, disp, counters, nullptr, 0.0f, out, stream );
		CUDA_CHECK( cudaStreamEndCapture( stream, &g ) );
		CUDA_CHECK( cudaGraphInstantiate( &ex, g, nullptr, nullptr, 0 ) );
		CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
		CUDA_CHECK( cudaStreamSynchronize( stream ) );
		CUDA_CHECK( cudaEventRecord( e0, stream ) );
		const int iters = 20;
		for ( int i = 0; i < iters; ++i ) CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
		CUDA_CHECK( cudaEventRecord( e1, stream ) );
		CUDA_CHECK( cudaEventSynchronize( e1 ) );
		float ms = 0.f;
		CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
		const double us = 1000.0 * ms / iters / SETS;
		std::printf( "  %2d local routes: %6.1f us per layer, %5.2f us per route, %4.0f GB/s\n", n, us, us / n, n * expert_bytes / ( us * 1e-6 ) / 1e9 );
		CUDA_CHECK( cudaGraphExecDestroy( ex ) );
		CUDA_CHECK( cudaGraphDestroy( g ) );
	}
	return 0;
}
