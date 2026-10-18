// The CPU NVFP4 expert (kernels/cpu/nvfp4_linear + runtime/cpu_experts) against the GPU NVFP4 MoE kernel on the same random weights and inputs.
// Needs a GPU, no model files.
#include "engine/config.h"
#include "engine/modules/nvfp4_moe.h"
#include "engine/runtime/cpu_experts.h"
#include "engine/kernels/cpu/nvfp4_linear.h"
#include "engine/kernels/cpu/cpu_util.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace engine::cpu_moe;
using namespace engine::kernels::cpu;
static std::mt19937 g_rng( 11 );
static float rnd() { return std::uniform_real_distribution<float>( -1.0f, 1.0f )( g_rng ); }
#define CU( x ) do { cudaError_t e_ = ( x ); if ( e_ != cudaSuccess ) { std::printf( "CUDA error %s at %d\n", cudaGetErrorString( e_ ), __LINE__ ); std::exit( 1 ); } } while ( 0 )

struct Fp4 { std::vector<uint8_t> codes, scales; float gs; };
static Fp4 make_fp4( int k, int n ) {
	Fp4 m; m.gs = 0.01f + 0.005f * rnd();
	m.codes.resize( static_cast<size_t>( n / 32 ) * ( k / 16 ) * 32 * 8 );
	m.scales.resize( static_cast<size_t>( n / 32 ) * ( k / 16 ) * 32 );
	for ( auto& c : m.codes ) c = static_cast<uint8_t>( g_rng() );
	for ( auto& c : m.scales ) c = static_cast<uint8_t>( 0x28 + g_rng() % 0x18 );
	return m;
}
template <class T> static T* up( const std::vector<T>& v ) { T* d; CU( cudaMalloc( &d, v.size() * sizeof( T ) ) ); CU( cudaMemcpy( d, v.data(), v.size() * sizeof( T ), cudaMemcpyHostToDevice ) ); return d; }

int main() {
	const int H = 2560, I = 640, E = 3, topk = 3;
	std::vector<Fp4> g, u, d;
	for ( int e = 0; e < E; ++e ) { g.push_back( make_fp4( H, I ) ); u.push_back( make_fp4( H, I ) ); d.push_back( make_fp4( I, H ) ); }
	std::vector<const uint8_t*> gc, gs, uc, us, dc, ds;
	std::vector<float> gg, ug, dg;
	for ( int e = 0; e < E; ++e ) {
		gc.push_back( up( g[e].codes ) ); gs.push_back( up( g[e].scales ) ); uc.push_back( up( u[e].codes ) ); us.push_back( up( u[e].scales ) );
		dc.push_back( up( d[e].codes ) ); ds.push_back( up( d[e].scales ) );
		gg.push_back( g[e].gs ); ug.push_back( u[e].gs ); dg.push_back( d[e].gs );
	}
	engine::modules::NVFP4ExpertTable t = {};
	t.gate_c = reinterpret_cast<const uint8_t* const*>( up( gc ) ); t.gate_s = reinterpret_cast<const uint8_t* const*>( up( gs ) );
	t.up_c = reinterpret_cast<const uint8_t* const*>( up( uc ) ); t.up_s = reinterpret_cast<const uint8_t* const*>( up( us ) );
	t.down_c = reinterpret_cast<const uint8_t* const*>( up( dc ) ); t.down_s = reinterpret_cast<const uint8_t* const*>( up( ds ) );
	t.gate_g = up( gg ); t.up_g = up( ug ); t.down_g = up( dg );
	t.num_local = E; t.min_expert = 0; t.hidden = H; t.inter = I;
	int fails = 0;
	for ( int rows : { 1, 4 } ) {
		std::vector<half> hx( static_cast<size_t>( rows ) * H ), hw( static_cast<size_t>( rows ) * topk );
		std::vector<int> idx( static_cast<size_t>( rows ) * topk );
		for ( auto& v : hx ) v = __float2half( rnd() * 2 );
		for ( int r = 0; r < rows; ++r ) for ( int k = 0; k < topk; ++k ) { idx[r * topk + k] = ( k + r ) % E; hw[r * topk + k] = __float2half( 0.3f + 0.1f * k ); }
		half *dx, *dw, *dh, *dout; int *didx, *ddisp, *dctr; float* dpart;
		CU( cudaMalloc( &dx, hx.size() * 2 ) ); CU( cudaMemcpy( dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice ) );
		CU( cudaMalloc( &dw, hw.size() * 2 ) ); CU( cudaMemcpy( dw, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice ) );
		didx = up( idx );
		CU( cudaMalloc( &dh, static_cast<size_t>( rows ) * topk * I * 2 ) ); CU( cudaMalloc( &dpart, static_cast<size_t>( rows ) * topk * H * 4 ) );
		CU( cudaMalloc( &ddisp, engine::modules::nvfp4_disp_ints( E ) * 4 ) ); CU( cudaMalloc( &dctr, ( H / 32 ) * 4 ) ); CU( cudaMemset( dctr, 0, ( H / 32 ) * 4 ) );
		CU( cudaMalloc( &dout, static_cast<size_t>( rows ) * H * 2 ) );
		engine::modules::NVFP4MoE::forward_rows( t, dx, didx, dw, rows, topk, dh, dpart, ddisp, dctr, nullptr, 0.0f, dout, nullptr );
		CU( cudaDeviceSynchronize() );
		std::vector<half> gpu( static_cast<size_t>( rows ) * H );
		CU( cudaMemcpy( gpu.data(), dout, gpu.size() * 2, cudaMemcpyDeviceToHost ) );
		// CPU
		std::vector<std::unique_ptr<Nvfp4Linear>> lin;
		std::vector<Expert> ex( E );
		for ( int e = 0; e < E; ++e ) {
			lin.emplace_back( new Nvfp4Linear( g[e].codes.data(), g[e].scales.data(), g[e].gs, H, I ) );
			lin.emplace_back( new Nvfp4Linear( u[e].codes.data(), u[e].scales.data(), u[e].gs, H, I ) );
			lin.emplace_back( new Nvfp4Linear( d[e].codes.data(), d[e].scales.data(), d[e].gs, I, H ) );
			ex[e].gate = lin[3 * e].get(); ex[e].up = lin[3 * e + 1].get(); ex[e].down = lin[3 * e + 2].get();
		}
		Pool pool( 4, {} );
		Scratch sc;
		std::vector<float> cpu( static_cast<size_t>( rows ) * H, 0.0f );
		for ( int r = 0; r < rows; ++r )
			for ( int k = 0; k < topk; ++k ) {
				std::vector<uint16_t> xh( H );
				std::memcpy( xh.data(), &hx[static_cast<size_t>( r ) * H], H * 2 );
				float w = __half2float( hw[r * topk + k] );
				std::vector<float> o( H, 0.0f );
				expert_forward( pool, ex[idx[r * topk + k]], 1, xh.data(), &w, o.data(), sc, 0.0f );
				for ( int c = 0; c < H; ++c ) cpu[static_cast<size_t>( r ) * H + c] += o[c];
			}
		double num = 0, den = 0, gnum = 0;
		for ( size_t i = 0; i < cpu.size(); ++i ) { const double a = __half2float( gpu[i] ); num += ( a - cpu[i] ) * ( a - cpu[i] ); den += a * a; }
		(void) gnum;
		const double rel = std::sqrt( num / ( den + 1e-30 ) );
		std::printf( "rows %d: CPU vs GPU relative error %.4g\n", rows, rel );
		if ( !( rel < 5e-3 ) ) ++fails;
	}
	std::printf( fails ? "%d FAILED\n" : "all passed\n", fails );
	return fails ? 1 : 0;
}
