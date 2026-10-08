#include <slang-cpp-prelude.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include "ref_cuda.h"

// Portable reference linear algebra (src/shaders/ref_gemm.slang, ref_trellis.slang).
//  (a) host target of the Slang kernels vs FP64 CPU references: dense GEMM, trellis dequant (all bit widths and codebooks),
//      block Hadamard, and the composed trellis Linear.
//  (b) with --gpu: the same kernels built for CUDA, and the host-target output, vs the engine's own reconstruct_slice and
//      LinearTrellis (forward + forward_recon) on identical packed data. Needs an idle GPU.

extern "C" {
void ref_gemm_f16( ComputeVaryingInput*, void*, void* );
void ref_trellis_reconstruct( ComputeVaryingInput*, void*, void* );
void ref_had128( ComputeVaryingInput*, void*, void* );
}

struct RefGemmParams { half* x; half* w; half* out_; int m; int k; int n; };
struct RefReconParams { uint16_t* codes; half* out_; int k_dim; int n_dim; int bits; int cb; };
struct RefHadParams { half* src; half* dst; half* pre; half* post; int cols; float r_scale; };

static uint32_t g_state = 4242u;
static uint32_t rnd() {
	g_state = g_state * 1664525u + 1013904223u;
	return g_state;
}
static float frand( float scale ) {
	return ( static_cast<int>( rnd() >> 16 ) % 2001 - 1000 ) / 1000.0f * scale;
}
static std::vector<half> make( size_t n, float scale ) {
	std::vector<half> v( n );
	for ( auto& x : v ) x = static_cast<half>( frand( scale ) );
	return v;
}
static std::vector<half> make_signs( size_t n ) {
	std::vector<half> v( n );
	for ( auto& x : v ) x = static_cast<half>( ( rnd() >> 20 ) & 1 ? 1.0f : -1.0f );
	return v;
}
static std::vector<uint16_t> make_codes( int k, int n, int bits ) {
	std::vector<uint16_t> c( static_cast<size_t>( k / 16 ) * ( n / 16 ) * 16 * bits );
	for ( auto& v : c ) v = static_cast<uint16_t>( rnd() >> 8 );
	return c;
}
static double h2d( half h ) { return static_cast<double>( static_cast<float>( h ) ); }

static void dispatch( void ( *fn )( ComputeVaryingInput*, void*, void* ), void* params, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1 ) {
	ComputeVaryingInput vi = {};
	vi.startGroupID = { 0, 0, 0 };
	vi.endGroupID = { gx, gy, gz };
	fn( &vi, params, nullptr );
}

// Tier 2 metrics: cosine similarity and max |d| relative to the reference peak
static bool compare( const char* name, const std::vector<double>& ref, const std::vector<double>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		dot += ref[i] * got[i]; na += ref[i] * ref[i]; nb += got[i] * got[i];
		maxd = std::fmax( maxd, std::fabs( ref[i] - got[i] ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = ref.size() == got.size() && cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-6;
	std::printf( "  %-34s cos %.7f max|d|/max %.3g  %s\n", name, cosv, maxd / ( maxabs + 1e-30 ), ok ? "PASS" : "FAIL" );
	return ok;
}
static std::vector<double> tod( const std::vector<half>& v ) {
	std::vector<double> d( v.size() );
	for ( size_t i = 0; i < v.size(); ++i ) d[i] = h2d( v[i] );
	return d;
}

// ---- FP64 references ----

static double half_bits_to_double( uint32_t h ) {
	const double sign = ( h & 0x8000u ) ? -1.0 : 1.0;
	const int e = static_cast<int>( ( h >> 10 ) & 31u );
	const double m = static_cast<double>( h & 1023u );
	if ( e == 0 ) return sign * m * std::ldexp( 1.0, -24 );
	return sign * ( 1.0 + m / 1024.0 ) * std::ldexp( 1.0, e - 15 );
}

// Codebook value of a 16-bit window in double precision, before FP16 rounding
static double codebook_ref( uint32_t w, int cb ) {
	if ( cb == 2 ) {
		const uint32_t x = w * 0x83DCD12Du;
		const uint32_t s = ( x & 255u ) + ( ( x >> 8 ) & 255u ) + ( ( x >> 16 ) & 255u ) + ( x >> 24 );
		return static_cast<double>( 1024u + s ) * half_bits_to_double( 0x1eeeu ) + half_bits_to_double( 0xc931u );
	}
	const uint32_t x = cb == 0 ? w * 89226354u + 64248484u : w * 0xCBAC1FEDu;
	const uint32_t y = ( x & 0x8fff8fffu ) ^ 0x3b603b60u;
	return half_bits_to_double( y & 0xffffu ) + half_bits_to_double( y >> 16 );
}

// Dequantized [k, n] weight in the rotated basis. Window extraction follows dq() (64-bit funnel shift over uint32 words) and the
// scatter follows the forward map of reconstruct_kernel, i.e. a different formulation from the Slang kernel's inverse map.
static std::vector<double> recon_ref( const std::vector<uint16_t>& codes, int k, int n, int bits, int cb ) {
	std::vector<double> w( static_cast<size_t>( k ) * n );
	const int nw = bits * 256 / 32;
	for ( int kt = 0; kt < k / 16; ++kt ) {
		for ( int nt = 0; nt < n / 16; ++nt ) {
			const uint16_t* tile = codes.data() + ( static_cast<size_t>( kt ) * ( n / 16 ) + nt ) * 16 * bits;
			auto word = [&]( int i ) { i %= nw; return static_cast<uint32_t>( tile[2 * i] ) | ( static_cast<uint32_t>( tile[2 * i + 1] ) << 16 ); };
			for ( int lane = 0; lane < 32; ++lane ) {
				for ( int i = 0; i < 8; ++i ) {
					const int p = lane * 8 + i;
					const int b0 = p * bits + bits - 16 + 256 * bits;
					const int b1 = b0 + 16;
					const int i0 = b0 / 32, i1 = ( b1 - 1 ) / 32;
					const int s0 = ( i1 + 1 ) * 32 - b1;
					const uint64_t merged = ( static_cast<uint64_t>( word( i0 ) ) << 32 ) | word( i1 );
					const uint32_t win = static_cast<uint32_t>( merged >> s0 ) & 0xffffu;
					const int row = ( lane % 4 ) * 2 + ( i & 1 ) + 8 * ( ( i >> 1 ) & 1 );
					const int col = 2 * ( lane / 8 + 4 * ( i >> 2 ) ) + ( ( lane >> 2 ) & 1 );
					w[static_cast<size_t>( kt * 16 + row ) * n + nt * 16 + col] = codebook_ref( win, cb );
				}
			}
		}
	}
	return w;
}

// Normalized Walsh-Hadamard over blocks of 128 (recursive butterfly, natural order) in double
static void had128_ref( std::vector<double>& v, int rows, int cols ) {
	for ( int r = 0; r < rows; ++r ) {
		for ( int b = 0; b < cols / 128; ++b ) {
			double* p = v.data() + static_cast<size_t>( r ) * cols + b * 128;
			for ( int len = 1; len < 128; len <<= 1 ) {
				for ( int i = 0; i < 128; i += 2 * len ) {
					for ( int j = i; j < i + len; ++j ) {
						const double a = p[j], c = p[j + len];
						p[j] = a + c;
						p[j + len] = a - c;
					}
				}
			}
			for ( int i = 0; i < 128; ++i ) p[i] /= std::sqrt( 128.0 );
		}
	}
}

static std::vector<double> gemm_ref( const std::vector<double>& x, const std::vector<double>& w, int m, int k, int n ) {
	std::vector<double> o( static_cast<size_t>( m ) * n, 0.0 );
	for ( int i = 0; i < m; ++i )
		for ( int p = 0; p < k; ++p )
			for ( int j = 0; j < n; ++j ) o[static_cast<size_t>( i ) * n + j] += x[static_cast<size_t>( i ) * k + p] * w[static_cast<size_t>( p ) * n + j];
	return o;
}

static std::vector<double> trellis_linear_ref( const std::vector<half>& x, const std::vector<double>& w, const std::vector<half>& suh,
	const std::vector<half>& svh, int m, int k, int n ) {
	std::vector<double> xh( x.size() );
	for ( int i = 0; i < m; ++i )
		for ( int p = 0; p < k; ++p ) xh[static_cast<size_t>( i ) * k + p] = h2d( x[static_cast<size_t>( i ) * k + p] ) * h2d( suh[p] );
	had128_ref( xh, m, k );
	auto y = gemm_ref( xh, w, m, k, n );
	had128_ref( y, m, n );
	for ( int i = 0; i < m; ++i )
		for ( int j = 0; j < n; ++j ) y[static_cast<size_t>( i ) * n + j] *= h2d( svh[j] );
	return y;
}

// ---- host-target kernel wrappers ----

static std::vector<half> host_gemm( const std::vector<half>& x, const std::vector<half>& w, int m, int k, int n ) {
	std::vector<half> out( static_cast<size_t>( m ) * n );
	RefGemmParams p{ const_cast<half*>( x.data() ), const_cast<half*>( w.data() ), out.data(), m, k, n };
	dispatch( ref_gemm_f16, &p, n, m );
	return out;
}
static std::vector<half> host_recon( const std::vector<uint16_t>& codes, int k, int n, int bits, int cb ) {
	std::vector<half> out( static_cast<size_t>( k ) * n );
	RefReconParams p{ const_cast<uint16_t*>( codes.data() ), out.data(), k, n, bits, cb };
	dispatch( ref_trellis_reconstruct, &p, static_cast<uint32_t>( ( static_cast<size_t>( k ) * n + 255 ) / 256 ) );
	return out;
}
static std::vector<half> host_had( const std::vector<half>& src, const half* pre, const half* post, int rows, int cols ) {
	std::vector<half> out( src.size() );
	RefHadParams p{ const_cast<half*>( src.data() ), out.data(), const_cast<half*>( pre ), const_cast<half*>( post ), cols, 0.08838834764831845f };
	dispatch( ref_had128, &p, cols / 128, rows );
	return out;
}
static std::vector<half> host_linear( const std::vector<half>& x, const std::vector<uint16_t>& codes, const std::vector<half>& suh,
	const std::vector<half>& svh, int m, int k, int n, int bits, int cb ) {
	auto xh = host_had( x, suh.data(), nullptr, m, k );
	auto w = host_recon( codes, k, n, bits, cb );
	auto y = host_gemm( xh, w, m, k, n );
	return host_had( y, nullptr, svh.data(), m, n );
}

// ---- GPU side ----

// Minimal CUDA runtime surface: the Slang C++ prelude and the CUDA headers define clashing vector types, so declare what is used
extern "C" {
int cudaMalloc( void**, size_t );
int cudaFree( void* );
int cudaMemcpy( void*, const void*, size_t, int );
int cudaDeviceSynchronize();
const char* cudaGetErrorString( int );
}
enum { cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };

#define CK( call ) do { int e_ = ( call ); if ( e_ != cudaSuccess ) { std::printf( "CUDA error %s at %s:%d\n", cudaGetErrorString( e_ ), __FILE__, __LINE__ ); std::exit( 2 ); } } while ( 0 )

template <typename T>
static T* upload( const std::vector<T>& v ) {
	void* d = nullptr;
	CK( cudaMalloc( &d, v.size() * sizeof( T ) ) );
	CK( cudaMemcpy( d, v.data(), v.size() * sizeof( T ), cudaMemcpyHostToDevice ) );
	return static_cast<T*>( d );
}
static std::vector<half> download( void* d, size_t n ) {
	std::vector<half> v( n );
	CK( cudaMemcpy( v.data(), d, n * sizeof( half ), cudaMemcpyDeviceToHost ) );
	return v;
}
static size_t count_mismatch( const std::vector<half>& a, const std::vector<half>& b ) {
	size_t bad = 0;
	for ( size_t i = 0; i < a.size(); ++i ) bad += std::memcmp( &a[i], &b[i], sizeof( half ) ) != 0;
	return bad;
}

static bool gpu_tests() {
	bool pass = true;
	std::printf( "=== reference kernels (CUDA + host target) vs engine LinearTrellis ===\n" );
	const int k = 256, n = 384;
	const int ms[] = { 1, 4, 40 };
	const int bit_list[] = { 3, 4, 5, 6 };
	for ( int cb = 2; cb >= 0; --cb ) {
		for ( int bits : bit_list ) {
			if ( cb != 2 && bits != 4 ) continue;
			std::printf( "[*] bits=%d cb=%d (%s)\n", bits, cb, cb == 2 ? "mul1" : cb == 1 ? "mcg" : "3inst" );
			auto codes = make_codes( k, n, bits );
			auto suh = make_signs( k ), svh = make_signs( n );
			uint16_t* d_codes = upload( codes );
			half* d_suh = upload( suh );
			half* d_svh = upload( svh );
			half *d_w_eng, *d_w_ref;
			CK( cudaMalloc( reinterpret_cast<void**>( &d_w_eng ), static_cast<size_t>( k ) * n * sizeof( half ) ) );
			CK( cudaMalloc( reinterpret_cast<void**>( &d_w_ref ), static_cast<size_t>( k ) * n * sizeof( half ) ) );
			ref_engine_reconstruct( d_codes, d_w_eng, k, n, bits, cb, nullptr );
			ref_trellis_reconstruct_cuda( d_codes, d_w_ref, k, n, bits, cb, nullptr );
			CK( cudaDeviceSynchronize() );
			auto w_eng = download( d_w_eng, static_cast<size_t>( k ) * n );
			auto w_ref = download( d_w_ref, static_cast<size_t>( k ) * n );
			auto w_host = host_recon( codes, k, n, bits, cb );
			const size_t bad_cuda = count_mismatch( w_eng, w_ref ), bad_host = count_mismatch( w_eng, w_host );
			std::printf( "  reconstruct bit-exact vs engine: slang-cuda %zu, host-target %zu mismatches of %d  %s\n",
				bad_cuda, bad_host, k * n, bad_cuda == 0 && bad_host == 0 ? "PASS" : "FAIL" );
			pass = pass && bad_cuda == 0 && bad_host == 0;
			for ( int m : ms ) {
				auto x = make( static_cast<size_t>( m ) * k, 1.0f );
				half* d_x = upload( x );
				half *d_out_eng, *d_out_ref, *d_xh, *d_y;
				CK( cudaMalloc( reinterpret_cast<void**>( &d_out_eng ), static_cast<size_t>( m ) * n * sizeof( half ) ) );
				CK( cudaMalloc( reinterpret_cast<void**>( &d_out_ref ), static_cast<size_t>( m ) * n * sizeof( half ) ) );
				CK( cudaMalloc( reinterpret_cast<void**>( &d_xh ), static_cast<size_t>( m ) * k * sizeof( half ) ) );
				CK( cudaMalloc( reinterpret_cast<void**>( &d_y ), static_cast<size_t>( m ) * n * sizeof( half ) ) );
				ref_trellis_linear_cuda( d_x, d_codes, d_suh, d_svh, d_out_ref, d_xh, d_w_ref, d_y, m, k, n, bits, cb, nullptr );
				CK( cudaDeviceSynchronize() );
				auto got = tod( download( d_out_ref, static_cast<size_t>( m ) * n ) );
				for ( int recon = 0; recon < 2; ++recon ) {
					ref_engine_trellis_linear( d_x, d_codes, d_suh, d_svh, d_out_eng, m, k, n, bits, cb, recon, nullptr );
					CK( cudaDeviceSynchronize() );
					auto eng = tod( download( d_out_eng, static_cast<size_t>( m ) * n ) );
					char name[96];
					std::snprintf( name, sizeof( name ), "linear m=%d ref-cuda vs engine %s", m, recon ? "recon" : "fused" );
					pass = compare( name, eng, got ) && pass;
				}
				auto host = tod( host_linear( x, codes, suh, svh, m, k, n, bits, cb ) );
				char name[96];
				std::snprintf( name, sizeof( name ), "linear m=%d ref-host vs ref-cuda", m );
				pass = compare( name, got, host ) && pass;
				cudaFree( d_x ); cudaFree( d_out_eng ); cudaFree( d_out_ref ); cudaFree( d_xh ); cudaFree( d_y );
			}
			cudaFree( d_codes ); cudaFree( d_suh ); cudaFree( d_svh ); cudaFree( d_w_eng ); cudaFree( d_w_ref );
		}
	}
	{ // dense GEMM on the GPU (odd sizes exercise the tile guards)
		const int m = 37, kk = 203, nn = 75;
		auto x = make( static_cast<size_t>( m ) * kk, 1.0f ), w = make( static_cast<size_t>( kk ) * nn, 1.0f );
		half *dx = upload( x ), *dw = upload( w ), *dout;
		CK( cudaMalloc( reinterpret_cast<void**>( &dout ), static_cast<size_t>( m ) * nn * sizeof( half ) ) );
		ref_gemm_f16_cuda( dx, dw, dout, m, kk, nn, nullptr );
		CK( cudaDeviceSynchronize() );
		pass = compare( "gemm 37x203x75 slang-cuda vs fp64", gemm_ref( tod( x ), tod( w ), m, kk, nn ), tod( download( dout, static_cast<size_t>( m ) * nn ) ) ) && pass;
		cudaFree( dx ); cudaFree( dw ); cudaFree( dout );
	}
	return pass;
}

int main( int argc, char** argv ) {
	bool pass = true;
	std::printf( "=== Slang host target vs FP64 references ===\n" );
	{ // dense GEMM, including m = 1 and sizes that are not multiples of 16
		const int ms[] = { 1, 5, 33 };
		for ( int m : ms ) {
			const int k = 200, n = 70;
			auto x = make( static_cast<size_t>( m ) * k, 1.0f ), w = make( static_cast<size_t>( k ) * n, 1.0f );
			char name[64];
			std::snprintf( name, sizeof( name ), "gemm m=%d k=%d n=%d", m, k, n );
			pass = compare( name, gemm_ref( tod( x ), tod( w ), m, k, n ), tod( host_gemm( x, w, m, k, n ) ) ) && pass;
		}
	}
	{ // trellis dequant: every bit width, all three codebooks; per-element error bound is the FP16 rounding of the result
		const int k = 64, n = 128;
		for ( int cb = 0; cb < 3; ++cb ) {
			for ( int bits = 1; bits <= 8; ++bits ) {
				auto codes = make_codes( k, n, bits );
				auto ref = recon_ref( codes, k, n, bits, cb );
				auto got = host_recon( codes, k, n, bits, cb );
				double worst = 0;
				bool ok = true;
				for ( size_t i = 0; i < ref.size(); ++i ) {
					const double d = std::fabs( ref[i] - h2d( got[i] ) );
					const double tol = std::fabs( ref[i] ) * 4.9e-4 + 1e-7;
					worst = std::fmax( worst, d / tol );
					ok = ok && d <= tol;
				}
				std::printf( "  %-34s worst err / half-ulp bound %.3f  %s\n", ( std::string( "dequant cb=" ) + std::to_string( cb ) + " bits=" + std::to_string( bits ) ).c_str(), worst, ok ? "PASS" : "FAIL" );
				pass = pass && ok;
			}
		}
	}
	{ // block Hadamard with pre and post scales
		const int rows = 3, cols = 384;
		auto src = make( static_cast<size_t>( rows ) * cols, 1.0f ), pre = make( cols, 1.0f ), post = make( cols, 1.0f );
		auto ref = tod( src );
		for ( int r = 0; r < rows; ++r ) for ( int c = 0; c < cols; ++c ) ref[static_cast<size_t>( r ) * cols + c] *= h2d( pre[c] );
		had128_ref( ref, rows, cols );
		for ( int r = 0; r < rows; ++r ) for ( int c = 0; c < cols; ++c ) ref[static_cast<size_t>( r ) * cols + c] *= h2d( post[c] );
		pass = compare( "had128 pre+post", ref, tod( host_had( src, pre.data(), post.data(), rows, cols ) ) ) && pass;
	}
	{ // composed trellis Linear
		const int k = 256, n = 384;
		const int bit_list[] = { 3, 4, 5, 6 };
		for ( int bits : bit_list ) {
			for ( int cb = 2; cb >= 1; --cb ) {
				const int m = 3;
				auto codes = make_codes( k, n, bits );
				auto x = make( static_cast<size_t>( m ) * k, 1.0f );
				auto suh = make_signs( k ), svh = make_signs( n );
				char name[64];
				std::snprintf( name, sizeof( name ), "trellis linear bits=%d cb=%d", bits, cb );
				pass = compare( name, trellis_linear_ref( x, recon_ref( codes, k, n, bits, cb ), suh, svh, m, k, n ),
					tod( host_linear( x, codes, suh, svh, m, k, n, bits, cb ) ) ) && pass;
			}
		}
	}
	bool gpu = false;
	for ( int i = 1; i < argc; ++i ) gpu = gpu || std::strcmp( argv[i], "--gpu" ) == 0;
	if ( gpu ) pass = gpu_tests() && pass;
	else std::printf( "(GPU comparison not run; pass --gpu)\n" );
	std::printf( "%s\n", pass ? "ALL PASS" : "FAILED" );
	return pass ? 0 : 1;
}
