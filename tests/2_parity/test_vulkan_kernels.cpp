// Slang SPIR-V kernels executed through the Vulkan backend on a real device, compared with FP64 references.
// Skips (exit 0) when no Vulkan device with the required features exists. Device: VOLTA_VK_DEVICE (default 0).
#include "engine/backend/vulkan/device.h"
#include "engine/backend/numeric.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace engine::backend;

#ifndef SLANG_SPV_DIR
	#error "SLANG_SPV_DIR must point at the generated .spv directory"
#endif

// Push-constant blocks: natural C layout == Slang's std430 emission (8-byte pointers, 4-byte scalars); mirrors test_slang_cpu.cpp
struct NormParams { uint64_t src, dst, weight; int hidden_size, has_weight; float eps, bias; };
struct AddParams { uint64_t hidden, residual; int n; };
struct SwigluParams { uint64_t gate, up, out_; int n; float limit; };
struct EmbedParams { uint64_t table, token_ids, out_; int hidden_size; };
struct RopeParams { uint64_t q, k, pos; int num_q_heads, num_kv_heads, head_dim, rot_dim, interleaved; float freq_base; };
struct ArgmaxParams { uint64_t logits, out_idx, out_val; int numel, batch_size, index_offset, row_stride, has_val; };
static_assert( sizeof( NormParams ) == 40 && sizeof( AddParams ) == 24 && sizeof( SwigluParams ) == 32, "push layout" );
static_assert( sizeof( EmbedParams ) == 32 && sizeof( RopeParams ) == 48 && sizeof( ArgmaxParams ) == 48, "push layout" );

static uint32_t g_state = 777u;
static float frand( float scale ) {
	g_state = g_state * 1664525u + 1013904223u;
	return ( static_cast<int>( g_state >> 16 ) % 2001 - 1000 ) / 1000.0f * scale;
}
static std::vector<half_t> make( size_t n, float scale ) {
	std::vector<half_t> v( n );
	for ( auto& x : v ) x = from_float_half( frand( scale ) );
	return v;
}
static double h2d( half_t h ) { return static_cast<double>( to_float( h ) ); }

// Device buffer holder: upload/download through the backend runtime
template<typename T> struct Dev {
	void* p = nullptr;
	size_t n = 0;
	explicit Dev( const std::vector<T>& v ) : n( v.size() ) {
		p = device_malloc( n * sizeof( T ) );
		memcpy_sync( p, v.data(), n * sizeof( T ), Copy::H2D );
	}
	explicit Dev( size_t count ) : n( count ) {
		p = device_malloc( n * sizeof( T ) );
		memset_sync( p, 0, ( ( n * sizeof( T ) + 3 ) / 4 ) * 4 );
	}
	std::vector<T> get() const {
		std::vector<T> v( n );
		memcpy_sync( v.data(), p, n * sizeof( T ), Copy::D2H );
		return v;
	}
	uint64_t addr() const { return reinterpret_cast<uint64_t>( p ); }
	~Dev() { device_free( p ); }
};

static vk::Pipeline* load( const char* name, uint32_t push_bytes ) {
	const std::string path = std::string( SLANG_SPV_DIR ) + "/" + name + ".spv";
	auto* p = vk::pipeline_create( path.c_str(), push_bytes );
	if ( !p ) std::printf( "  cannot create pipeline %s\n", path.c_str() );
	return p;
}

static bool check( const char* name, const std::vector<double>& ref, const std::vector<half_t>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const double b = h2d( got[i] );
		dot += ref[i] * b; na += ref[i] * ref[i]; nb += b * b;
		maxd = std::fmax( maxd, std::fabs( ref[i] - b ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-22s cos %.7f max|d| %.3g (max %.3g)  %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}

int main() {
	if ( !vk::available() ) {
		std::printf( "test_vulkan_kernels: no Vulkan device with BDA/Float16/Int64, skipping\n" );
		return 0;
	}
	std::printf( "=== Slang SPIR-V through Vulkan: %s (push constants max %u B) ===\n", vk::device_name(), vk::max_push_constants_bytes() );
	bool pass = true;

	{ // conversion helpers
		bool ok = to_float( from_float_half( 1.5f ) ) == 1.5f && from_float_half( 65520.0f ).bits == 0x7c00 && from_float_half( 6e-8f ).bits == 1
			&& to_float( from_float_bf16( 3.0f ) ) == 3.0f && to_float( half_t{ 1 } ) == 5.9604645e-8f;
		std::printf( "  %-22s %s\n", "half/bf16 helpers", ok ? "PASS" : "FAIL" );
		pass = ok && pass;
	}

	{ // rms_norm
		auto* pl = load( "rms_norm", sizeof( NormParams ) );
		const int rows = 4, hs = 1000;
		auto x = make( rows * hs, 1.0f ), w = make( hs, 0.5f );
		Dev<half_t> dx( x ), dw( w ), dy( size_t( rows ) * hs );
		NormParams p{ dx.addr(), dy.addr(), dw.addr(), hs, 1, 1e-6f, 1.0f };
		vk::dispatch( nullptr, pl, &p, rows );
		stream_sync( nullptr );
		std::vector<double> ref( rows * hs );
		for ( int r = 0; r < rows; ++r ) {
			double ss = 0;
			for ( int i = 0; i < hs; ++i ) ss += h2d( x[r * hs + i] ) * h2d( x[r * hs + i] );
			const double inv = 1.0 / std::sqrt( ss / hs + 1e-6 );
			for ( int i = 0; i < hs; ++i ) ref[r * hs + i] = h2d( x[r * hs + i] ) * ( h2d( w[i] ) + 1.0 ) * inv;
		}
		pass = check( "rms_norm", ref, dy.get() ) && pass;
		vk::pipeline_destroy( pl );
	}
	{ // add_residual
		auto* pl = load( "add_residual", sizeof( AddParams ) );
		const int n = 100003;
		auto a = make( n, 2.0f ), b = make( n, 2.0f );
		Dev<half_t> da( a ), db( b );
		AddParams p{ da.addr(), db.addr(), n };
		vk::dispatch( nullptr, pl, &p, ( n + 255 ) / 256 );
		stream_sync( nullptr );
		std::vector<double> ref( n );
		for ( int i = 0; i < n; ++i ) ref[i] = h2d( a[i] ) + h2d( b[i] );
		pass = check( "add_residual", ref, da.get() ) && pass;
		vk::pipeline_destroy( pl );
	}
	{ // swiglu
		auto* pl = load( "swiglu", sizeof( SwigluParams ) );
		const int n = 5000;
		auto g = make( n, 3.0f ), u = make( n, 2.0f );
		Dev<half_t> dg( g ), du( u ), dout( static_cast<size_t>( n ) );
		SwigluParams p{ dg.addr(), du.addr(), dout.addr(), n, 0.0f };
		vk::dispatch( nullptr, pl, &p, ( n + 255 ) / 256 );
		stream_sync( nullptr );
		std::vector<double> ref( n );
		for ( int i = 0; i < n; ++i ) { const double gv = h2d( g[i] ); ref[i] = gv / ( 1.0 + std::exp( -gv ) ) * h2d( u[i] ); }
		pass = check( "swiglu", ref, dout.get() ) && pass;
		vk::pipeline_destroy( pl );
	}
	{ // embed_tokens (BF16 table)
		auto* pl = load( "embed_tokens", sizeof( EmbedParams ) );
		const int vocab = 64, hs = 300, rows = 5;
		std::vector<uint16_t> table( size_t( vocab ) * hs );
		for ( auto& t : table ) t = from_float_bf16( frand( 2.0f ) ).bits;
		std::vector<int> ids = { 3, 63, 0, 17, 42 };
		Dev<uint16_t> dt( table );
		Dev<int> di( ids );
		Dev<half_t> dout( static_cast<size_t>( rows * hs ) );
		EmbedParams p{ dt.addr(), di.addr(), dout.addr(), hs };
		vk::dispatch( nullptr, pl, &p, ( hs + 255 ) / 256, rows );
		stream_sync( nullptr );
		std::vector<double> ref( size_t( rows ) * hs );
		for ( int r = 0; r < rows; ++r ) for ( int c = 0; c < hs; ++c ) ref[size_t( r ) * hs + c] = to_float( bf16_t{ table[size_t( ids[r] ) * hs + c] } );
		pass = check( "embed_tokens", ref, dout.get() ) && pass;
		vk::pipeline_destroy( pl );
	}
	for ( int il = 0; il < 2; ++il ) { // rope
		auto* pl = load( "rope", sizeof( RopeParams ) );
		const int rows = 3, qh = 4, kh = 2, hd = 64, rot = 32;
		auto q = make( rows * qh * hd, 1.0f ), k = make( rows * kh * hd, 1.0f );
		std::vector<int> pos = { 1000, 1001, 1002 };
		Dev<half_t> dq( q ), dk( k );
		Dev<int> dp( pos );
		RopeParams p{ dq.addr(), dk.addr(), dp.addr(), qh, kh, hd, rot, il, 1000000.0f };
		vk::dispatch( nullptr, pl, &p, rows );
		stream_sync( nullptr );
		std::vector<double> ref( q.size() );
		for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < qh; ++h ) for ( int c = 0; c < hd; ++c ) {
			const size_t o = ( static_cast<size_t>( r ) * qh + h ) * hd;
			double v = h2d( q[o + c] );
			if ( c < rot ) {
				const int i = il ? c >> 1 : ( c < rot / 2 ? c : c - rot / 2 );
				const double th = pos[r] * std::pow( 1000000.0, -2.0 * i / rot );
				const int p0 = il ? 2 * i : i, p1 = il ? 2 * i + 1 : i + rot / 2;
				const double a = h2d( q[o + p0] ), b = h2d( q[o + p1] );
				v = ( c == p0 ) ? a * std::cos( th ) - b * std::sin( th ) : a * std::sin( th ) + b * std::cos( th );
			}
			ref[o + c] = v;
		}
		pass = check( il ? "rope interleaved q" : "rope contiguous q", ref, dq.get() ) && pass;
		vk::pipeline_destroy( pl );
	}
	{ // batched_argmax
		auto* pl = load( "batched_argmax", sizeof( ArgmaxParams ) );
		const int batch = 3, numel = 900, stride = 1024, offset = 50;
		auto logits = make( batch * stride, 3.0f );
		Dev<half_t> dl( logits );
		Dev<int> di( static_cast<size_t>( batch ) );
		Dev<float> dv( static_cast<size_t>( batch ) );
		ArgmaxParams p{ dl.addr(), di.addr(), dv.addr(), numel, batch, offset, stride, 1 };
		vk::dispatch( nullptr, pl, &p, batch );
		stream_sync( nullptr );
		auto idx = di.get();
		auto val = dv.get();
		bool ok = true;
		for ( int r = 0; r < batch; ++r ) {
			int bi = -1; float bv = -1e30f;
			for ( int i = 0; i < numel; ++i ) { const float v = to_float( logits[r * stride + i] ); if ( v > bv ) { bv = v; bi = i; } }
			ok = ok && idx[r] == bi + offset && val[r] == bv;
		}
		std::printf( "  %-22s %s\n", "batched_argmax", ok ? "PASS" : "FAIL" );
		pass = ok && pass;
		vk::pipeline_destroy( pl );
	}
	{ // pre-recorded command buffer of 64 dispatches, replayed (informational)
		auto* pl = load( "add_residual", sizeof( AddParams ) );
		const int n = 1 << 20, reps = 20;
		std::vector<half_t> a( n, from_float_half( 0.0f ) ), b( n, from_float_half( 0.25f ) );
		Dev<half_t> da( a ), db( b );
		AddParams p{ da.addr(), db.addr(), n };
		stream_t s = stream_create();
		graph_capture_begin( s );
		for ( int i = 0; i < 64; ++i ) vk::dispatch( s, pl, &p, ( n + 255 ) / 256 );
		graph_t g = graph_capture_end( s );
		graph_exec_t e = graph_instantiate( g );
		graph_launch( e, s ); // warm
		stream_sync( s );
		const auto t0 = std::chrono::steady_clock::now();
		for ( int r = 0; r < reps; ++r ) graph_launch( e, s );
		stream_sync( s );
		const double us = std::chrono::duration<double, std::micro>( std::chrono::steady_clock::now() - t0 ).count();
		auto out = da.get();
		const double expect = 0.25 * 64 * ( reps + 1 );
		const bool ok = std::fabs( h2d( out[0] ) - expect ) < 1e-3 && std::fabs( h2d( out[n - 1] ) - expect ) < 1e-3;
		std::printf( "  %-22s %zu nodes, %.1f us/replay (%.2f us/dispatch)  %s\n", "replay 64 dispatches", graph_node_count( g ), us / reps, us / reps / 64, ok ? "PASS" : "FAIL" );
		pass = ok && pass;
		graph_exec_destroy( e );
		graph_destroy( g );
		stream_destroy( s );
		vk::pipeline_destroy( pl );
	}

	std::printf( pass ? "ALL PASS\n" : "FAILURES\n" );
	return pass ? 0 : 1;
}
