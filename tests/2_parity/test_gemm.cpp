// Gemm::tn and Gemm::nn (the engine's FP16 tensor-core GEMM, weights stored [n, k] or [k, n]) against an FP64 host product of the same FP16
// operands: odd sizes on every edge (rows, columns, the k tail), both output types, beta, a strided output, every tile size and split-k
#include "engine/modules/gemm.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace backend = engine::backend;
using backend::half_t;

int main() {
	std::printf( "=== Gemm::tn / Gemm::nn against an FP64 reference ===\n" );
	backend::set_device( 0 );
	std::mt19937 rng( 7 );
	std::normal_distribution<float> nd( 0.0f, 1.0f );
	struct Case { int m, n, k, ldc_pad; bool f32; float beta; };
	const Case cases[] = { { 1, 64, 64, 0, false, 0.0f }, { 20, 288, 4096, 0, false, 0.0f }, { 37, 130, 72, 0, true, 0.0f }, { 64, 2560, 1152, 0, false, 0.0f },
		{ 100, 333, 520, 6, true, 1.0f }, { 129, 1000, 2048, 0, false, 0.0f }, { 512, 768, 640, 2, false, 0.5f }, { 300, 257, 136, 0, true, 0.0f },
		{ 160, 512, 3072, 0, true, 0.0f }, { 384, 1024, 1024, 0, false, 0.0f } };
	int fails = 0;
	for ( int form = 0; form < 2; ++form )
	for ( const Case& c : cases ) {
		const bool nn = form == 1;   // weights [k, n]
		if ( nn && c.n % 8 != 0 ) continue;
		const int ldc = c.n + c.ldc_pad;
		std::vector<half_t> a( static_cast<size_t>( c.m ) * c.k ), w( static_cast<size_t>( c.n ) * c.k );
		for ( auto& v : a ) v = backend::from_float_half( nd( rng ) );
		for ( auto& v : w ) v = backend::from_float_half( nd( rng ) * 0.05f );
		std::vector<float> init( static_cast<size_t>( c.m ) * ldc );
		for ( auto& v : init ) v = nd( rng );
		std::vector<double> ref( static_cast<size_t>( c.m ) * ldc );
		for ( int i = 0; i < c.m; ++i ) {
			for ( int j = 0; j < c.n; ++j ) {
				double s = 0.0;
				for ( int t = 0; t < c.k; ++t ) {
					const half_t wv = nn ? w[static_cast<size_t>( t ) * c.n + j] : w[static_cast<size_t>( j ) * c.k + t];
					s += static_cast<double>( backend::to_float( a[static_cast<size_t>( i ) * c.k + t] ) ) * backend::to_float( wv );
				}
				const float base = c.f32 ? init[static_cast<size_t>( i ) * ldc + j] : backend::to_float( backend::from_float_half( init[static_cast<size_t>( i ) * ldc + j] ) );
				ref[static_cast<size_t>( i ) * ldc + j] = s + c.beta * base;
			}
		}
		half_t* da = static_cast<half_t*>( backend::device_malloc( a.size() * sizeof( half_t ) ) );
		half_t* dw = static_cast<half_t*>( backend::device_malloc( w.size() * sizeof( half_t ) ) );
		const size_t ob = static_cast<size_t>( c.m ) * ldc * ( c.f32 ? sizeof( float ) : sizeof( half_t ) );
		void* dout = backend::device_malloc( ob );
		backend::memcpy_sync( da, a.data(), a.size() * sizeof( half_t ), backend::Copy::H2D );
		backend::memcpy_sync( dw, w.data(), w.size() * sizeof( half_t ), backend::Copy::H2D );
		std::vector<half_t> init_h( init.size() );
		for ( size_t i = 0; i < init.size(); ++i ) init_h[i] = backend::from_float_half( init[i] );
		backend::memcpy_sync( dout, c.f32 ? static_cast<const void*>( init.data() ) : static_cast<const void*>( init_h.data() ), ob, backend::Copy::H2D );
		if ( nn ) engine::modules::Gemm::nn( da, c.k, dw, c.n, dout, ldc, c.f32, c.m, c.n, c.k, c.beta );
		else engine::modules::Gemm::tn( da, c.k, dw, c.k, dout, ldc, c.f32, c.m, c.n, c.k, c.beta );
		backend::device_sync();
		std::vector<float> got( static_cast<size_t>( c.m ) * ldc );
		if ( c.f32 ) backend::memcpy_sync( got.data(), dout, ob, backend::Copy::D2H );
		else {
			std::vector<half_t> gh( got.size() );
			backend::memcpy_sync( gh.data(), dout, ob, backend::Copy::D2H );
			for ( size_t i = 0; i < gh.size(); ++i ) got[i] = backend::to_float( gh[i] );
		}
		double err = 0.0, norm = 0.0;
		bool pad_ok = true;
		for ( int i = 0; i < c.m; ++i ) {
			for ( int j = 0; j < ldc; ++j ) {
				const size_t o = static_cast<size_t>( i ) * ldc + j;
				if ( j >= c.n ) { pad_ok = pad_ok && got[o] == ( c.f32 ? init[o] : backend::to_float( init_h[o] ) ); continue; }
				err += ( got[o] - ref[o] ) * ( got[o] - ref[o] );
				norm += ref[o] * ref[o];
			}
		}
		const double rel = std::sqrt( err / std::max( norm, 1e-30 ) );
		const bool ok = rel < ( c.f32 ? 1e-4 : 2e-3 ) && pad_ok;
		std::printf( "  %s m %4d n %4d k %4d ldc %4d %s beta %.1f: rel rms %.2e%s  %s\n", nn ? "nn" : "tn", c.m, c.n, c.k, ldc, c.f32 ? "fp32" : "fp16", c.beta, rel,
			pad_ok ? "" : " (wrote past n)", ok ? "PASS" : "FAIL" );
		fails += ok ? 0 : 1;
		backend::device_free( da );
		backend::device_free( dw );
		backend::device_free( dout );
	}
	std::printf( "%s\n", fails ? "[-] Gemm FAIL" : "[+] Gemm PASS" );
	return fails ? 1 : 0;
}
