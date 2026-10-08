// Vulkan configuration smoke test: module entry points (the ones tagged Portable in tier_table.inc) run through the MODULE API on a
// Vulkan device (TensorViews over device_malloc'd memory, upload / download through the runtime layer) and compared with FP64
// references. Skips (exit 0) when no Vulkan device with the required features exists. Device: VOLTA_VK_DEVICE (default 0).
#include "engine/backend/numeric.h"
#include "engine/backend/runtime.h"
#include "engine/backend/vulkan/device.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/modules/attention.h"
#include "engine/modules/embedding.h"
#include "engine/modules/gdn.h"
#include "engine/modules/linear.h"
#include "engine/modules/mhc.h"
#include "engine/modules/norm.h"
#include "engine/modules/rope.h"
#include "engine/modules/swiglu.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace engine;
using backend::Copy;
using backend::half_t;
using pod::DType;

static uint32_t g_state = 4242u;
static uint32_t rnd() {
	g_state = g_state * 1664525u + 1013904223u;
	return g_state;
}
static float frand( float scale ) { return ( static_cast<int>( rnd() >> 16 ) % 2001 - 1000 ) / 1000.0f * scale; }
static std::vector<half_t> make( size_t n, float scale ) {
	std::vector<half_t> v( n );
	for ( auto& x : v ) x = backend::from_float_half( frand( scale ) );
	return v;
}
static double d( half_t h ) { return static_cast<double>( backend::to_float( h ) ); }

// Device buffer: upload / download through the runtime layer
template<typename T> struct Dev {
	void* p = nullptr;
	size_t n = 0;
	explicit Dev( const std::vector<T>& v ) : n( v.size() ) {
		p = backend::device_malloc( n * sizeof( T ) );
		backend::memcpy_sync( p, v.data(), n * sizeof( T ), Copy::H2D );
	}
	explicit Dev( size_t count ) : n( count ) {
		p = backend::device_malloc( n * sizeof( T ) );
		backend::memset_sync( p, 0, ( ( n * sizeof( T ) + 3 ) / 4 ) * 4 );
	}
	std::vector<T> get() const {
		std::vector<T> v( n );
		backend::memcpy_sync( v.data(), p, n * sizeof( T ), Copy::D2H );
		return v;
	}
	~Dev() { backend::device_free( p ); }
	T* ptr() const { return static_cast<T*>( p ); }
};

static pod::TensorView view2( void* p, DType t, int r, int c ) { return pod::make_tensor_2d( p, t, r, c ); }

static bool check( const char* name, const std::vector<double>& ref, const std::vector<half_t>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const double b = d( got[i] );
		dot += ref[i] * b; na += ref[i] * ref[i]; nb += b * b;
		maxd = std::fmax( maxd, std::fabs( ref[i] - b ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = ref.size() == got.size() && std::isfinite( cosv ) && cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-34s cos %.7f max|d| %.3g (max %.3g)  %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}
static bool check_exact( const char* name, bool ok ) {
	std::printf( "  %-34s %s\n", name, ok ? "PASS" : "FAIL" );
	return ok;
}

int main() {
	if ( !backend::vk::available() ) {
		std::printf( "test_vulkan_smoke: no Vulkan device with BDA/Float16/Int64, skipping\n" );
		return 0;
	}
	std::printf( "=== Vulkan module API smoke: %s ===\n", backend::vk::device_name() );
	bool pass = true;

	{ // RMSNorm::forward, with weight (bias 1) and without
		const int rows = 5, hs = 1000;
		auto x = make( rows * hs, 1.0f ), w = make( hs, 0.5f );
		Dev<half_t> dx( x ), dw( w ), dy( size_t( rows ) * hs ), dy2( size_t( rows ) * hs );
		auto tx = view2( dx.p, DType::FP16, rows, hs ), tw = pod::make_tensor_1d( dw.p, DType::FP16, hs ), ty = view2( dy.p, DType::FP16, rows, hs );
		modules::RMSNorm::forward( tx, tw, ty, 1e-6f, 1.0f );
		auto ty2 = view2( dy2.p, DType::FP16, rows, hs );
		pod::TensorView none = {};
		modules::RMSNorm::forward( tx, none, ty2, 1e-6f, 0.0f );
		backend::stream_sync( nullptr );
		std::vector<double> ref( rows * hs ), ref2( rows * hs );
		for ( int r = 0; r < rows; ++r ) {
			double ss = 0;
			for ( int i = 0; i < hs; ++i ) ss += d( x[r * hs + i] ) * d( x[r * hs + i] );
			const double inv = 1.0 / std::sqrt( ss / hs + 1e-6 );
			for ( int i = 0; i < hs; ++i ) {
				ref[r * hs + i] = d( x[r * hs + i] ) * ( d( w[i] ) + 1.0 ) * inv;
				ref2[r * hs + i] = d( x[r * hs + i] ) * inv;
			}
		}
		pass = check( "RMSNorm::forward", ref, dy.get() ) && pass;
		pass = check( "RMSNorm::forward (no weight)", ref2, dy2.get() ) && pass;
	}
	{ // RMSNorm::forward_heads
		const int rows = 3, heads = 6, hd = 128;
		auto x = make( rows * heads * hd, 1.0f ), w = make( hd, 0.5f );
		Dev<half_t> dx( x ), dw( w );
		auto tx = view2( dx.p, DType::FP16, rows, heads * hd ), tw = pod::make_tensor_1d( dw.p, DType::FP16, hd );
		modules::RMSNorm::forward_heads( tx, tw, heads, hd, 1e-6f, 1.0f );
		backend::stream_sync( nullptr );
		std::vector<double> ref( x.size() );
		for ( int g = 0; g < rows * heads; ++g ) {
			double ss = 0;
			for ( int i = 0; i < hd; ++i ) ss += d( x[g * hd + i] ) * d( x[g * hd + i] );
			const double inv = 1.0 / std::sqrt( ss / hd + 1e-6 );
			for ( int i = 0; i < hd; ++i ) ref[g * hd + i] = d( x[g * hd + i] ) * ( d( w[i] ) + 1.0 ) * inv;
		}
		pass = check( "RMSNorm::forward_heads", ref, dx.get() ) && pass;
	}
	{ // Elementwise::add_residual
		const int n = 100003;
		auto a = make( n, 2.0f ), b = make( n, 2.0f );
		Dev<half_t> da( a ), db( b );
		auto ta = pod::make_tensor_1d( da.p, DType::FP16, n ), tb = pod::make_tensor_1d( db.p, DType::FP16, n );
		modules::Elementwise::add_residual( ta, tb );
		backend::stream_sync( nullptr );
		std::vector<double> ref( n );
		for ( int i = 0; i < n; ++i ) ref[i] = d( a[i] ) + d( b[i] );
		pass = check( "Elementwise::add_residual", ref, da.get() ) && pass;
	}
	{ // argmax, batched_argmax, batched_argmax_range
		const int batch = 4, vocab = 5000;
		auto lg = make( batch * vocab, 4.0f );
		for ( int r = 0; r < batch; ++r ) lg[r * vocab + 17 + 700 * r] = backend::from_float_half( 9.0f + r );
		lg[2 * vocab + 4999] = backend::from_float_half( 20.0f );   // beyond the range window of the range test
		Dev<half_t> dl( lg );
		Dev<int> di{ size_t( batch ) }, di2{ size_t( batch ) }, di3{ size_t( batch ) }, di4{ size_t( 1 ) };
		Dev<float> dv{ size_t( batch ) };
		auto tl = view2( dl.p, DType::FP16, batch, vocab );
		modules::Elementwise::batched_argmax( tl, di.ptr(), vocab, batch );
		modules::Elementwise::batched_argmax_range( tl, di2.ptr(), dv.ptr(), 4000, batch, 100 );
		auto t1 = view2( dl.p, DType::FP16, 1, vocab );
		modules::Elementwise::argmax( t1, di4.ptr(), vocab );
		backend::stream_sync( nullptr );
		const auto i1 = di.get(), i2 = di2.get(), i4 = di4.get();
		const auto v = dv.get();
		bool ok = true, ok2 = true;
		for ( int r = 0; r < batch; ++r ) {
			int best = 0, best_r = 0;
			for ( int i = 1; i < vocab; ++i ) if ( d( lg[r * vocab + i] ) > d( lg[r * vocab + best] ) ) best = i;
			for ( int i = 1; i < 4000; ++i ) if ( d( lg[r * vocab + i] ) > d( lg[r * vocab + best_r] ) ) best_r = i;
			ok = ok && i1[r] == best;
			ok2 = ok2 && i2[r] == best_r + 100 && v[r] == static_cast<float>( d( lg[r * vocab + best_r] ) );
		}
		int b0 = 0;
		for ( int i = 1; i < vocab; ++i ) if ( d( lg[i] ) > d( lg[b0] ) ) b0 = i;
		pass = check_exact( "Elementwise::batched_argmax", ok ) && pass;
		pass = check_exact( "Elementwise::batched_argmax_range", ok2 ) && pass;
		pass = check_exact( "Elementwise::argmax", i4[0] == b0 ) && pass;
	}
	{ // SwiGLU::forward / forward_fused
		const int rows = 7, inter = 700, n = rows * inter;
		auto g = make( n, 3.0f ), u = make( n, 2.0f ), gu = make( size_t( rows ) * 2 * inter, 3.0f );
		Dev<half_t> dg( g ), du( u ), dgu( gu ), o1{ size_t( n ) }, o2{ size_t( n ) };
		auto tg = view2( dg.p, DType::FP16, rows, inter ), tu = view2( du.p, DType::FP16, rows, inter ), to1 = view2( o1.p, DType::FP16, rows, inter );
		auto tgu = view2( dgu.p, DType::FP16, rows, 2 * inter ), to2 = view2( o2.p, DType::FP16, rows, inter );
		modules::SwiGLU::forward( tg, tu, to1 );
		modules::SwiGLU::forward_fused( tgu, to2 );
		backend::stream_sync( nullptr );
		std::vector<double> ref( n ), ref2( n );
		for ( int i = 0; i < n; ++i ) {
			const double gv = d( g[i] );
			ref[i] = gv / ( 1.0 + std::exp( -gv ) ) * d( u[i] );
			const int r = i / inter, c = i % inter;
			const double a = d( gu[r * 2 * inter + c] ), b = d( gu[r * 2 * inter + inter + c] );
			ref2[i] = a / ( 1.0 + std::exp( -a ) ) * b;
		}
		pass = check( "SwiGLU::forward", ref, o1.get() ) && pass;
		pass = check( "SwiGLU::forward_fused", ref2, o2.get() ) && pass;
	}
	{ // RoPE::forward, both pairings; BatchRows in device memory
		for ( int interleaved = 0; interleaved < 2; ++interleaved ) {
			const int rows = 6, hq = 8, hkv = 2, hd = 128, rot = 64;
			const float base = 10000.0f;
			auto q = make( size_t( rows ) * hq * hd, 1.0f ), k = make( size_t( rows ) * hkv * hd, 1.0f );
			int slots[6] = { 0, 0, 0, 1, 1, 2 }, pos[6] = { 3, 4, 5, 100, 101, 7000 };
			pod::BatchRows br = pod::BatchRows::batched( rows, slots, pos );
			Dev<pod::BatchRows> dr( std::vector<pod::BatchRows>{ br } );
			Dev<half_t> dq( q ), dk( k );
			auto tq = view2( dq.p, DType::FP16, rows, hq * hd ), tk = view2( dk.p, DType::FP16, rows, hkv * hd );
			modules::RoPE::forward( tq, tk, hq, hkv, hd, rot, dr.ptr(), rows, base, nullptr, interleaved != 0 );
			backend::stream_sync( nullptr );
			auto rot_ref = [&]( const std::vector<half_t>& src, int heads ) {
				std::vector<double> o( src.size() );
				for ( size_t i = 0; i < src.size(); ++i ) o[i] = d( src[i] );
				for ( int r = 0; r < rows; ++r ) {
					for ( int h = 0; h < heads; ++h ) {
						for ( int pr = 0; pr < rot / 2; ++pr ) {
							const double th = pos[r] * std::pow( static_cast<double>( base ), -2.0 * pr / rot );
							const int i0 = interleaved ? 2 * pr : pr, i1 = interleaved ? 2 * pr + 1 : pr + rot / 2;
							const size_t off = ( static_cast<size_t>( r ) * heads + h ) * hd;
							const double a = d( src[off + i0] ), b = d( src[off + i1] );
							o[off + i0] = a * std::cos( th ) - b * std::sin( th );
							o[off + i1] = a * std::sin( th ) + b * std::cos( th );
						}
					}
				}
				return o;
			};
			pass = check( interleaved ? "RoPE::forward q (interleaved)" : "RoPE::forward q", rot_ref( q, hq ), dq.get() ) && pass;
			pass = check( interleaved ? "RoPE::forward k (interleaved)" : "RoPE::forward k", rot_ref( k, hkv ), dk.get() ) && pass;
		}
	}
	{ // Embedding::forward (BF16 table)
		const int vocab = 64, hs = 300, rows = 5;
		std::vector<uint16_t> table( size_t( vocab ) * hs );
		for ( auto& t : table ) t = backend::from_float_bf16( frand( 2.0f ) ).bits;
		std::vector<int> ids = { 3, 63, 0, 17, 17 };
		Dev<uint16_t> dt( table );
		Dev<int> di( ids );
		Dev<half_t> out( size_t( rows ) * hs );
		auto tt = view2( dt.p, DType::BF16, vocab, hs ), to = view2( out.p, DType::FP16, rows, hs );
		modules::Embedding::forward( tt, di.ptr(), rows, to );
		backend::stream_sync( nullptr );
		std::vector<double> ref( size_t( rows ) * hs );
		for ( int r = 0; r < rows; ++r )
			for ( int c = 0; c < hs; ++c ) ref[size_t( r ) * hs + c] = backend::to_float( backend::bf16_t{ table[size_t( ids[r] ) * hs + c] } );
		pass = check( "Embedding::forward", ref, out.get() ) && pass;
	}
	{ // GDN::l2_norm_heads (q and k slices of a packed row) and GDN::gated_rmsnorm
		const int rows = 3, hk = 4, hd = 128, stride = 6 * hd * 2;   // q|k|v packed: 2*hk heads normalized, remaining columns untouched
		auto x = make( size_t( rows ) * stride, 1.0f );
		Dev<half_t> dx( x );
		auto tx = view2( dx.p, DType::FP16, rows, stride );
		modules::GDN::l2_norm_heads( tx, hk, hd, nullptr, true, rows, stride );
		backend::stream_sync( nullptr );
		std::vector<double> ref( x.size() );
		for ( int r = 0; r < rows; ++r ) {
			for ( int b = 0; b < stride / hd; ++b ) {
				const size_t off = size_t( r ) * stride + size_t( b ) * hd;
				double ss = 0;
				for ( int i = 0; i < hd; ++i ) ss += d( x[off + i] ) * d( x[off + i] );
				double inv = b < 2 * hk ? 1.0 / std::sqrt( ss + 1e-6 ) : 1.0;
				if ( b < hk ) inv /= std::sqrt( static_cast<double>( hd ) );
				for ( int i = 0; i < hd; ++i ) ref[off + i] = d( x[off + i] ) * inv;
			}
		}
		pass = check( "GDN::l2_norm_heads", ref, dx.get() ) && pass;
	}
	{
		const int rows = 4, hv = 8, hd = 128;
		for ( int sig = 0; sig < 2; ++sig ) {
			auto x = make( size_t( rows ) * hv * hd, 1.0f ), z = make( x.size(), 2.0f ), w = make( hd, 1.0f );
			Dev<half_t> dx( x ), dz( z ), dw( w );
			auto tx = view2( dx.p, DType::FP16, rows, hv * hd ), tz = view2( dz.p, DType::FP16, rows, hv * hd ), tw = pod::make_tensor_1d( dw.p, DType::FP16, hd );
			modules::GDN::gated_rmsnorm( tx, tz, tw, hv, hd, nullptr, sig != 0, rows );
			backend::stream_sync( nullptr );
			std::vector<double> ref( x.size() );
			for ( int g = 0; g < rows * hv; ++g ) {
				double ss = 0;
				for ( int i = 0; i < hd; ++i ) ss += d( x[g * hd + i] ) * d( x[g * hd + i] );
				const double inv = 1.0 / std::sqrt( ss / hd + 1e-6 );
				for ( int i = 0; i < hd; ++i ) {
					const double zv = d( z[g * hd + i] );
					const double gate = sig ? 1.0 / ( 1.0 + std::exp( -zv ) ) : zv / ( 1.0 + std::exp( -zv ) );
					ref[g * hd + i] = d( x[g * hd + i] ) * inv * d( w[i] ) * gate;
				}
			}
			pass = check( sig ? "GDN::gated_rmsnorm (sigmoid)" : "GDN::gated_rmsnorm (silu)", ref, dx.get() ) && pass;
		}
	}
	{ // Attention::store_kv_cache (FP16)
		const int count = 4, hkv = 2, hd = 64, max_seq = 32, layer = 1, tot = hkv * hd;
		const size_t slot_stride = size_t( 3 ) * max_seq * tot;
		auto k = make( size_t( count ) * tot, 1.0f ), v = make( size_t( count ) * tot, 1.0f );
		int slots[4] = { 0, 0, 1, 2 }, pos[4] = { 5, 6, 0, 31 };
		pod::BatchRows br = pod::BatchRows::batched( count, slots, pos );
		Dev<pod::BatchRows> dr( std::vector<pod::BatchRows>{ br } );
		Dev<half_t> dk( k ), dv( v ), ck( slot_stride * 3 ), cv( slot_stride * 3 );
		auto tk = view2( dk.p, DType::FP16, count, tot ), tv = view2( dv.p, DType::FP16, count, tot );
		modules::Attention::store_kv_cache( pod::kv_ref_f16( ck.p, cv.p ), tk, tv, slot_stride, layer, dr.ptr(), count, hkv, hd, max_seq );
		backend::stream_sync( nullptr );
		std::vector<double> rk( ck.n, 0.0 ), rv( cv.n, 0.0 );
		for ( int m = 0; m < count; ++m ) {
			for ( int e = 0; e < tot; ++e ) {
				const size_t dst = size_t( slots[m] ) * slot_stride + ( size_t( layer ) * max_seq + pos[m] ) * tot + e;
				rk[dst] = d( k[size_t( m ) * tot + e] );
				rv[dst] = d( v[size_t( m ) * tot + e] );
			}
		}
		pass = check( "Attention::store_kv_cache K", rk, ck.get() ) && pass;
		pass = check( "Attention::store_kv_cache V", rv, cv.get() ) && pass;
	}
	{ // HyperConnection::init_streams, HyperHead::mean
		const int rows = 3, hs = 700;
		auto x = make( size_t( rows ) * hs, 1.0f );
		Dev<half_t> dx( x ), out( size_t( rows ) * hs );
		Dev<float> st( size_t( rows ) * 4 * hs );
		auto tx = view2( dx.p, DType::FP16, rows, hs ), tst = pod::make_tensor_3d( st.p, DType::FP32, rows, 4, hs ), to = view2( out.p, DType::FP16, rows, hs );
		modules::HyperConnection::init_streams( tx, tst, hs, nullptr, rows );
		modules::HyperHead::mean( tst, to, hs, nullptr, rows );
		backend::stream_sync( nullptr );
		const auto s = st.get();
		bool ok = true;
		for ( int r = 0; r < rows; ++r )
			for ( int h = 0; h < 4; ++h )
				for ( int c = 0; c < hs; ++c ) ok = ok && s[( size_t( r ) * 4 + h ) * hs + c] == static_cast<float>( d( x[size_t( r ) * hs + c] ) );
		pass = check_exact( "HyperConnection::init_streams", ok ) && pass;
		std::vector<double> ref( x.size() );
		for ( size_t i = 0; i < ref.size(); ++i ) ref[i] = d( x[i] );
		pass = check( "HyperHead::mean", ref, out.get() ) && pass;
	}
	{ // LinearPlainGEMV::forward (ref_gemm_nt_f16, weight [n, k]), ragged sizes
		const int m = 5, k = 300, n = 130;
		auto x = make( size_t( m ) * k, 1.0f ), w = make( size_t( k ) * n, 0.2f );
		Dev<half_t> dx( x ), dw( w ), out( size_t( m ) * n );
		modules::LinearPlainGEMV::forward( dw.ptr(), dx.ptr(), out.ptr(), m, n, k, nullptr );
		backend::stream_sync( nullptr );
		std::vector<double> ref( size_t( m ) * n, 0.0 );
		for ( int i = 0; i < m; ++i )
			for ( int p = 0; p < k; ++p )
				for ( int j = 0; j < n; ++j ) ref[size_t( i ) * n + j] += d( x[size_t( i ) * k + p] ) * d( w[size_t( j ) * k + p] );
		pass = check( "LinearPlainGEMV::forward", ref, out.get() ) && pass;
	}

	backend::device_sync();
	std::printf( "%s\n", pass ? "test_vulkan_smoke: PASSED" : "test_vulkan_smoke: FAILED" );
	return pass ? 0 : 1;
}
