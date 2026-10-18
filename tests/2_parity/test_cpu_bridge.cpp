// The GPU <-> CPU expert handoff (engine/modules/cpu_experts.h + engine/runtime/cpu_expert_service.h) with synthetic NVFP4 experts: routes to
// CPU-owned experts must leave the route list, their expert outputs must arrive in the layer output (also from a CUDA graph, over many replays),
// and layers without CPU routes must not wait. Needs a GPU, no model files.
#include "engine/config.h"
#include "engine/modules/cpu_experts.h"
#include "engine/runtime/cpu_expert_service.h"
#include "engine/kernels/cpu/nvfp4_linear.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace engine::cpu_moe;
using namespace engine::kernels::cpu;
using engine::modules::CpuExperts;

static std::mt19937 g_rng( 7 );
static float rnd() { return std::uniform_real_distribution<float>( -1.0f, 1.0f )( g_rng ); }
static int g_fail = 0;
#define CHECK( cond, ... ) do { if ( !( cond ) ) { ++g_fail; std::printf( "FAIL: " __VA_ARGS__ ); std::printf( "\n" ); } } while ( 0 )
#define CU( x ) do { cudaError_t e_ = ( x ); if ( e_ != cudaSuccess ) { std::printf( "CUDA error %s at %d\n", cudaGetErrorString( e_ ), __LINE__ ); std::exit( 1 ); } } while ( 0 )

struct Fp4 { std::vector<uint8_t> codes, scales; float gs; };
static Fp4 make_fp4( int k, int n ) {
	Fp4 m; m.gs = 0.02f;
	m.codes.resize( static_cast<size_t>( n / 32 ) * ( k / 16 ) * 32 * 8 );
	m.scales.resize( static_cast<size_t>( n / 32 ) * ( k / 16 ) * 32 );
	for ( auto& c : m.codes ) c = static_cast<uint8_t>( g_rng() );
	for ( auto& c : m.scales ) c = static_cast<uint8_t>( 0x28 + g_rng() % 0x20 );
	return m;
}

int main() {
	const int H = 512, I = 256, E = 8, topk = 4, rows = 3, layers = 3, min_e = 16;   // this "rank" owns experts 16..23
	std::vector<int> cpus;
	for ( int c = 10; c < 16; ++c ) cpus.push_back( c );
	Service& svc = Service::get();
	svc.start( cpus );
	Mailbox* mb = svc.mailbox( 0, H, 10.0f );
	// experts 1, 4, 6 of each layer live on the CPU; a private copy of the same linears is the reference
	const int owned[3] = { 1, 4, 6 };
	std::vector<std::unique_ptr<Fp4>> weights;   // the linears read the codes in place: keep them alive
	std::vector<std::unique_ptr<Nvfp4Linear>> ref;
	std::vector<std::vector<Expert>> ref_ex( layers );
	for ( int l = 0; l < layers; ++l )
		for ( int e : owned ) {
			weights.emplace_back( new Fp4( make_fp4( H, I ) ) ); weights.emplace_back( new Fp4( make_fp4( H, I ) ) ); weights.emplace_back( new Fp4( make_fp4( I, H ) ) );
			const Fp4 &g = *weights[weights.size() - 3], &u = *weights[weights.size() - 2], &d = *weights[weights.size() - 1];
			svc.add_expert( 0, l, min_e + e, std::unique_ptr<Linear>( new Nvfp4Linear( g.codes.data(), g.scales.data(), g.gs, H, I ) ),
				std::unique_ptr<Linear>( new Nvfp4Linear( u.codes.data(), u.scales.data(), u.gs, H, I ) ), std::unique_ptr<Linear>( new Nvfp4Linear( d.codes.data(), d.scales.data(), d.gs, I, H ) ) );
			ref.emplace_back( new Nvfp4Linear( g.codes.data(), g.scales.data(), g.gs, H, I ) );
			ref.emplace_back( new Nvfp4Linear( u.codes.data(), u.scales.data(), u.gs, H, I ) );
			ref.emplace_back( new Nvfp4Linear( d.codes.data(), d.scales.data(), d.gs, I, H ) );
			Expert ex; ex.gate = ref[ref.size() - 3].get(); ex.up = ref[ref.size() - 2].get(); ex.down = ref[ref.size() - 1].get();
			ref_ex[l].push_back( ex );
		}
	uint8_t h_owner[E] = {};
	for ( int e : owned ) h_owner[e] = 1;
	uint8_t* d_owner; CU( cudaMalloc( &d_owner, E ) ); CU( cudaMemcpy( d_owner, h_owner, E, cudaMemcpyHostToDevice ) );
	half *d_x, *d_rw, *d_out; int* d_sel; uint32_t* d_heat; uint64_t* d_ctl;
	CU( cudaMalloc( &d_x, rows * H * 2 ) ); CU( cudaMalloc( &d_rw, rows * topk * 2 ) ); CU( cudaMalloc( &d_out, rows * H * 2 ) );
	CU( cudaMalloc( &d_sel, rows * topk * 4 ) ); CU( cudaMalloc( &d_heat, E * 4 ) ); CU( cudaMalloc( &d_ctl, 16 ) );
	CU( cudaMemset( d_ctl, 0, 16 ) ); CU( cudaMemset( d_heat, 0, E * 4 ) );
	cudaStream_t st; CU( cudaStreamCreate( &st ) );

	auto fill = [&]( std::vector<half>& x, std::vector<int>& sel, std::vector<half>& rw, bool force_none ) {
		for ( auto& v : x ) v = __float2half( rnd() );
		for ( int r = 0; r < rows; ++r ) {
			int chosen[topk];
			for ( int k = 0; k < topk; ++k ) {
				int e;
				bool dup;
				do {
					e = force_none ? 0 : static_cast<int>( g_rng() % ( E + 2 ) ) - 1;   // -1 and E are outside this rank's range
					if ( force_none ) e = ( k % 2 == 0 ) ? 0 : 2;                        // experts 0 and 2: not owned
					dup = false;
					for ( int j = 0; j < k; ++j ) dup |= chosen[j] == e;
				} while ( dup && !force_none );
				chosen[k] = e;
				sel[r * topk + k] = min_e + e;
				rw[r * topk + k] = __float2half( 0.2f + 0.1f * k );
			}
		}
	};
	std::vector<half> hx( rows * H ), hrw( rows * topk ), hout( rows * H );
	std::vector<int> hsel( rows * topk ), sel_after( rows * topk );

	// ---- eager: post, wait_add per layer ----
	int checked = 0;
	for ( int iter = 0; iter < 300; ++iter ) {
		const int layer = iter % layers;
		fill( hx, hsel, hrw, iter % 7 == 3 );
		CU( cudaMemcpy( d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice ) );
		CU( cudaMemcpy( d_rw, hrw.data(), hrw.size() * 2, cudaMemcpyHostToDevice ) );
		CU( cudaMemcpy( d_sel, hsel.data(), hsel.size() * 4, cudaMemcpyHostToDevice ) );
		CU( cudaMemset( d_out, 0, rows * H * 2 ) );
		CpuExperts::post( d_x, d_sel, d_rw, rows, topk, H, min_e, min_e + E, d_owner, 1.0f, d_heat, mb, d_ctl, layer, st );
		CpuExperts::wait_add( d_out, rows, H, mb, d_ctl, st );
		CU( cudaStreamSynchronize( st ) );
		CU( cudaMemcpy( hout.data(), d_out, hout.size() * 2, cudaMemcpyDeviceToHost ) );
		CU( cudaMemcpy( sel_after.data(), d_sel, sel_after.size() * 4, cudaMemcpyDeviceToHost ) );
		// reference
		std::vector<float> ref_out( rows * H, 0.0f );
		bool any = false;
		for ( int r = 0; r < rows; ++r )
			for ( int k = 0; k < topk; ++k ) {
				const int e = hsel[r * topk + k] - min_e;
				const bool own = e >= 0 && e < E && h_owner[e];
				CHECK( sel_after[r * topk + k] == ( own ? -1 : hsel[r * topk + k] ), "iter %d route (%d,%d): sel %d -> %d (owned %d)", iter, r, k, hsel[r * topk + k], sel_after[r * topk + k], own );
				if ( !own ) continue;
				any = true;
				int idx = 0;
				for ( int j = 0; j < 3; ++j ) if ( owned[j] == e ) idx = j;
				std::vector<uint16_t> xh( H );
				std::memcpy( xh.data(), &hx[r * H], H * 2 );
				float w = __half2float( hrw[r * topk + k] );
				Pool p1( 1, {} );
				Scratch sc;
				expert_forward( p1, ref_ex[layer][idx], 1, xh.data(), &w, &ref_out[r * H], sc );
			}
		double num = 0, den = 0;
		for ( int i = 0; i < rows * H; ++i ) { const double d = __half2float( hout[i] ) - ref_out[i]; num += d * d; den += ref_out[i] * ref_out[i]; }
		const double rel = std::sqrt( num / ( den + 1e-30 ) );
		CHECK( any ? rel < 2e-3 : den == 0 && num == 0, "iter %d layer %d: relative error %.3g (cpu routes %d)", iter, layer, rel, any );
		++checked;
	}
	std::printf( "eager: %d layers checked\n", checked );

	// ---- frac 0.5: the CPU takes the last floor( d / 2 ) of the d distinct owned experts; the rest stay in the route list ----
	for ( int iter = 0; iter < 60; ++iter ) {
		fill( hx, hsel, hrw, false );
		CU( cudaMemcpy( d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice ) );
		CU( cudaMemcpy( d_rw, hrw.data(), hrw.size() * 2, cudaMemcpyHostToDevice ) );
		CU( cudaMemcpy( d_sel, hsel.data(), hsel.size() * 4, cudaMemcpyHostToDevice ) );
		CU( cudaMemset( d_out, 0, rows * H * 2 ) );
		CpuExperts::post( d_x, d_sel, d_rw, rows, topk, H, min_e, min_e + E, d_owner, 0.5f, d_heat, mb, d_ctl, 0, st );
		CpuExperts::wait_add( d_out, rows, H, mb, d_ctl, st );
		CU( cudaStreamSynchronize( st ) );
		CU( cudaMemcpy( sel_after.data(), d_sel, sel_after.size() * 4, cudaMemcpyDeviceToHost ) );
		std::vector<int> distinct, taken;
		for ( int i = 0; i < rows * topk; ++i ) {
			const int e = hsel[i] - min_e;
			if ( e >= 0 && e < E && h_owner[e] && std::find( distinct.begin(), distinct.end(), hsel[i] ) == distinct.end() ) distinct.push_back( hsel[i] );
			if ( sel_after[i] == -1 && hsel[i] >= min_e && hsel[i] < min_e + E ) taken.push_back( hsel[i] );
		}
		std::vector<int> want( distinct.rbegin(), distinct.rbegin() + static_cast<long>( distinct.size() / 2 ) );
		for ( int i = 0; i < rows * topk; ++i ) {
			const bool w = std::find( want.begin(), want.end(), hsel[i] ) != want.end();
			CHECK( sel_after[i] == ( w ? -1 : hsel[i] ), "frac iter %d route %d: %d -> %d (wanted %d)", iter, i, hsel[i], sel_after[i], w );
		}
	}
	std::printf( "fraction: checked\n" );

	// ---- graph: post + wait_add captured once, replayed with new data ----
	cudaGraph_t graph; cudaGraphExec_t exec;
	CU( cudaStreamBeginCapture( st, cudaStreamCaptureModeThreadLocal ) );
	CpuExperts::post( d_x, d_sel, d_rw, rows, topk, H, min_e, min_e + E, d_owner, 1.0f, d_heat, mb, d_ctl, 1, st );
	CpuExperts::wait_add( d_out, rows, H, mb, d_ctl, st );
	CU( cudaStreamEndCapture( st, &graph ) );
	CU( cudaGraphInstantiate( &exec, graph, nullptr, nullptr, 0 ) );
	auto t0 = std::chrono::steady_clock::now();
	int graph_ok = 0;
	for ( int iter = 0; iter < 200; ++iter ) {
		fill( hx, hsel, hrw, false );
		CU( cudaMemcpy( d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice ) );
		CU( cudaMemcpy( d_rw, hrw.data(), hrw.size() * 2, cudaMemcpyHostToDevice ) );
		CU( cudaMemcpy( d_sel, hsel.data(), hsel.size() * 4, cudaMemcpyHostToDevice ) );
		CU( cudaMemset( d_out, 0, rows * H * 2 ) );
		CU( cudaGraphLaunch( exec, st ) );
		CU( cudaStreamSynchronize( st ) );
		CU( cudaMemcpy( hout.data(), d_out, hout.size() * 2, cudaMemcpyDeviceToHost ) );
		std::vector<float> ref_out( rows * H, 0.0f );
		for ( int r = 0; r < rows; ++r )
			for ( int k = 0; k < topk; ++k ) {
				const int e = hsel[r * topk + k] - min_e;
				if ( !( e >= 0 && e < E && h_owner[e] ) ) continue;
				int idx = 0;
				for ( int j = 0; j < 3; ++j ) if ( owned[j] == e ) idx = j;
				std::vector<uint16_t> xh( H );
				std::memcpy( xh.data(), &hx[r * H], H * 2 );
				float w = __half2float( hrw[r * topk + k] );
				Pool p1( 1, {} );
				Scratch sc;
				expert_forward( p1, ref_ex[1][idx], 1, xh.data(), &w, &ref_out[r * H], sc );
			}
		double num = 0, den = 0;
		for ( int i = 0; i < rows * H; ++i ) { const double d = __half2float( hout[i] ) - ref_out[i]; num += d * d; den += ref_out[i] * ref_out[i]; }
		CHECK( std::sqrt( num / ( den + 1e-30 ) ) < 2e-3, "graph iter %d: relative error %.3g", iter, std::sqrt( num / ( den + 1e-30 ) ) );
		++graph_ok;
	}
	std::printf( "graph: %d replays checked in %.1f ms\n", graph_ok, std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - t0 ).count() );
	CHECK( mb->err == 0, "device timeout flagged" );
	std::printf( g_fail ? "%d FAILED\n" : "all passed\n", g_fail );
	return g_fail ? 1 : 0;
}
