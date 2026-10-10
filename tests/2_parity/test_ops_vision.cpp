// Op-level cross-backend golden test for the vision-tower operators (Vision::*): norms, patch embedding, interpolated position table, rotary with
// biases, bidirectional attention, bias / activation / residual helpers, the merger's gather and the dense GEMM. ONE source builds in the CUDA
// configuration (--dump runs the real kernels and writes golden files) and in the CPU / Vulkan ones (--check recomputes and compares).
//   test_ops_vision --dump <dir> | --check <dir>
#include "golden_io.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/modules/vision.h"
#include <algorithm>
#include <cmath>
#include <vector>

using namespace engine;
using backend::half_t;
using modules::Vision;

namespace {

GoldenArgs* g_args = nullptr;
int g_fail = 0;
int g_cases = 0;
std::vector<void*> g_allocs;

template <typename T>
T* up( const std::vector<T>& v ) {
	T* p = static_cast<T*>( backend::device_malloc( std::max<size_t>( v.size() * sizeof( T ), 16 ) ) );
	g_allocs.push_back( p );
	if ( !v.empty() ) backend::memcpy_sync( p, v.data(), v.size() * sizeof( T ), backend::Copy::H2D );
	return p;
}

template <typename T>
T* zeros( size_t n ) {
	T* p = static_cast<T*>( backend::device_malloc( std::max<size_t>( n * sizeof( T ), 16 ) ) );
	g_allocs.push_back( p );
	backend::memset_sync( p, 0, std::max<size_t>( n * sizeof( T ), 16 ) );
	return p;
}

std::vector<half_t> rh( GoldenRng& r, size_t n, float scale, float offset = 0.0f ) {
	std::vector<half_t> v( n );
	for ( auto& x : v ) x = backend::from_float_half( offset + r.uniform( scale ) );
	return v;
}

std::vector<float> rf( GoldenRng& r, size_t n, float scale ) {
	std::vector<float> v( n );
	for ( auto& x : v ) x = r.uniform( scale );
	return v;
}

std::vector<float> dl( const half_t* d, size_t n ) {
	backend::device_sync();
	std::vector<half_t> h( n );
	backend::memcpy_sync( h.data(), d, n * sizeof( half_t ), backend::Copy::D2H );
	std::vector<float> f( n );
	for ( size_t i = 0; i < n; ++i ) f[i] = backend::to_float( h[i] );
	return f;
}

std::vector<float> dlf( const float* d, size_t n ) {
	backend::device_sync();
	std::vector<float> f( n );
	backend::memcpy_sync( f.data(), d, n * sizeof( float ), backend::Copy::D2H );
	return f;
}

void emit( const char* name, const std::vector<float>& v ) {
	++g_cases;
	if ( g_args->dump ) {
		if ( !golden_write( g_args->dir, name, v.data(), v.size() * sizeof( float ) ) ) {
			std::printf( "  %-28s cannot write golden file  FAIL\n", name );
			++g_fail;
		}
		else {
			std::printf( "  %-28s dumped (%zu floats)\n", name, v.size() );
		}
		return;
	}
	std::vector<uint8_t> bytes;
	if ( !golden_read( g_args->dir, name, bytes ) ) {
		std::printf( "  %-28s golden file missing  FAIL\n", name );
		++g_fail;
		return;
	}
	std::vector<float> ref( bytes.size() / sizeof( float ) );
	std::memcpy( ref.data(), bytes.data(), ref.size() * sizeof( float ) );
	if ( !golden_compare( name, ref, v ) ) ++g_fail;
}

} // namespace

int main( int argc, char** argv ) {
	GoldenArgs args( argc, argv );
	g_args = &args;
	if ( !args.dump && !args.check ) {
		std::printf( "usage: test_ops_vision --dump <dir> | --check <dir>\n" );
		return 2;
	}
	backend::set_device( 0 );
	std::printf( "test_ops_vision (%s, dir %s)\n", args.dump ? "dump" : "check", args.dir.c_str() );
	GoldenRng rng( 9301 );

	{ // f32 -> f16, position table interpolation, patch embedding
		const size_t n = 1000;
		float* in = up( rf( rng, n, 3.0f ) );
		half_t* out = zeros<half_t>( n );
		Vision::f32_to_f16( in, out, n );
		emit( "vis_f32_to_f16", dl( out, n ) );

		const int gh = 6, gw = 8, merge = 2, side = 7, dim = 40;
		half_t* table = up( rh( rng, static_cast<size_t>( side ) * side * dim, 1.0f ) );
		half_t* pos = zeros<half_t>( static_cast<size_t>( gh ) * gw * dim );
		Vision::pos_interp( table, pos, gh, gw, merge, side, dim );
		emit( "vis_pos_interp", dl( pos, static_cast<size_t>( gh ) * gw * dim ) );

		const int m = 24, nn = 96, k = 48;
		half_t* patches = up( rh( rng, static_cast<size_t>( m ) * k, 1.0f ) );
		half_t* w = up( rh( rng, static_cast<size_t>( nn ) * k, 0.2f ) );
		half_t* bias = up( rh( rng, nn, 0.5f ) );
		half_t* add = up( rh( rng, static_cast<size_t>( m ) * nn, 0.5f ) );
		float* pe = zeros<float>( static_cast<size_t>( m ) * nn );
		Vision::patch_embed( patches, w, bias, add, pe, m, nn, k );
		emit( "vis_patch_embed", dlf( pe, static_cast<size_t>( m ) * nn ) );
	}

	{ // norms
		const int m = 12, n = 80;
		float* x = up( rf( rng, static_cast<size_t>( m ) * n, 2.0f ) );
		half_t* w = up( rh( rng, n, 0.5f, 1.0f ) );
		half_t* b = up( rh( rng, n, 0.3f ) );
		half_t* o1 = zeros<half_t>( static_cast<size_t>( m ) * n );
		Vision::rms_norm( x, w, o1, m, n, 1e-6f );
		emit( "vis_rms_norm", dl( o1, static_cast<size_t>( m ) * n ) );
		half_t* o2 = zeros<half_t>( static_cast<size_t>( m ) * n );
		Vision::layer_norm( x, w, b, o2, m, n, 1e-6f );
		emit( "vis_layer_norm", dl( o2, static_cast<size_t>( m ) * n ) );
		half_t* o3 = zeros<half_t>( static_cast<size_t>( m ) * n );
		Vision::layer_norm_merge( x, w, b, o3, m, n, 4, 1e-6f );
		emit( "vis_layer_norm_merge", dl( o3, static_cast<size_t>( m ) * n ) );
		half_t* xh = up( rh( rng, static_cast<size_t>( m ) * n, 2.0f ) );
		half_t* o4 = zeros<half_t>( static_cast<size_t>( m ) * n );
		Vision::layer_norm_half( xh, w, b, o4, m, n, 1e-6f );
		emit( "vis_layer_norm_half", dl( o4, static_cast<size_t>( m ) * n ) );
	}

	{ // rotary with biases: the 64-wide norm form and the generic form
		const int m = 20, heads = 4;
		for ( int form = 0; form < 2; ++form ) {
			const int hd = form == 0 ? 64 : 72, width = heads * hd;
			half_t* q = up( rh( rng, static_cast<size_t>( m ) * width, 1.0f ) );
			half_t* k = up( rh( rng, static_cast<size_t>( m ) * width, 1.0f ) );
			half_t* v = up( rh( rng, static_cast<size_t>( m ) * width, 1.0f ) );
			half_t* bq = up( rh( rng, width, 0.3f ) );
			half_t* bk = up( rh( rng, width, 0.3f ) );
			half_t* bv = up( rh( rng, width, 0.3f ) );
			float* cs = up( rf( rng, static_cast<size_t>( m ) * hd, 1.0f ) );
			float* sn = up( rf( rng, static_cast<size_t>( m ) * hd, 1.0f ) );
			if ( form == 0 ) {
				half_t* qn = up( rh( rng, 64, 0.3f, 1.0f ) );
				half_t* kn = up( rh( rng, 64, 0.3f, 1.0f ) );
				Vision::qkv_bias_norm_rope( q, k, v, bq, bk, bv, qn, kn, cs, sn, m, heads, hd, 1e-6f );
				emit( "vis_qkv_norm_rope_q", dl( q, static_cast<size_t>( m ) * width ) );
				emit( "vis_qkv_norm_rope_k", dl( k, static_cast<size_t>( m ) * width ) );
				emit( "vis_qkv_norm_rope_v", dl( v, static_cast<size_t>( m ) * width ) );
			}
			else {
				Vision::qkv_bias_rope( q, k, v, bq, bk, bv, cs, sn, m, heads, hd );
				emit( "vis_qkv_rope_q", dl( q, static_cast<size_t>( m ) * width ) );
				emit( "vis_qkv_rope_k", dl( k, static_cast<size_t>( m ) * width ) );
				emit( "vis_qkv_rope_v", dl( v, static_cast<size_t>( m ) * width ) );
			}
		}
	}

	{ // attention (both head widths the towers use), bidirectional over all rows
		const int m = 37, heads = 4;
		for ( int hd : { 64, 72 } ) {
			const int width = heads * hd;
			half_t* q = up( rh( rng, static_cast<size_t>( m ) * width, 1.0f ) );
			half_t* k = up( rh( rng, static_cast<size_t>( m ) * width, 1.0f ) );
			half_t* v = up( rh( rng, static_cast<size_t>( m ) * width, 1.0f ) );
			half_t* o = zeros<half_t>( static_cast<size_t>( m ) * width );
			Vision::attention( q, k, v, o, m, heads, hd, 1.0f / std::sqrt( static_cast<float>( hd ) ) );
			char nm[48];
			std::snprintf( nm, sizeof( nm ), "vis_attention_%d", hd );
			emit( nm, dl( o, static_cast<size_t>( m ) * width ) );
		}
	}

	{ // merger gather, dense GEMM, bias / activation / residual
		const int tokens = 5, n = 24, group = 4;
		half_t* h = up( rh( rng, static_cast<size_t>( tokens ) * group * n, 1.0f ) );
		half_t* og = zeros<half_t>( static_cast<size_t>( tokens ) * group * n );
		Vision::merge_gather( h, og, tokens, n, group );
		emit( "vis_merge_gather", dl( og, static_cast<size_t>( tokens ) * group * n ) );

		const int m = 19, nn = 56, k = 64;
		half_t* a = up( rh( rng, static_cast<size_t>( m ) * k, 1.0f ) );
		half_t* w = up( rh( rng, static_cast<size_t>( nn ) * k, 0.2f ) );
		half_t* og2 = zeros<half_t>( static_cast<size_t>( m ) * nn );
		Vision::dense_gemm( a, w, og2, m, nn, k );
		emit( "vis_dense_gemm", dl( og2, static_cast<size_t>( m ) * nn ) );

		half_t* bias = up( rh( rng, nn, 0.5f ) );
		for ( int act : { Vision::ACT_NONE, Vision::ACT_GELU_TANH, Vision::ACT_GELU_ERF } ) {
			half_t* x = up( rh( rng, static_cast<size_t>( m ) * nn, 2.0f ) );
			Vision::bias_act( x, bias, m, nn, act );
			char nm[48];
			std::snprintf( nm, sizeof( nm ), "vis_bias_act_%d", act );
			emit( nm, dl( x, static_cast<size_t>( m ) * nn ) );
		}
		float* hidden = up( rf( rng, static_cast<size_t>( m ) * nn, 1.0f ) );
		half_t* y = up( rh( rng, static_cast<size_t>( m ) * nn, 1.0f ) );
		Vision::bias_residual( hidden, y, bias, m, nn );
		emit( "vis_bias_residual", dlf( hidden, static_cast<size_t>( m ) * nn ) );
	}

	backend::device_sync();
	for ( void* p : g_allocs ) backend::device_free( p );
	std::printf( "%d outputs, %d failures\n", g_cases, g_fail );
	return g_fail == 0 ? 0 : 1;
}
