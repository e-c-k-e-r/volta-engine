// CPU backend smoke test (ENGINE_BACKEND=CPU): module entry points run through the Slang host kernels and match FP64
// references (tier 2: cosine >= 0.9999, max |d| <= 0.005 * max + 1e-3). Also checks that a CUDA/SM70-only entry point
// aborts with the named "CPU backend: ... is not implemented" message.
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/core/arena.h"
#include "engine/core/batch.h"
#include "engine/core/graph_cache.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/rope.h"
#include "engine/modules/embedding.h"
#include "engine/modules/expert_tier.h"
#include "engine/modules/trellis.h"
#include "engine/modules/gdn.h"
#include "engine/modules/linear.h"
#include "engine/modules/mhc.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

using namespace engine;
using backend::half_t;

static uint32_t g_state = 12345u;
static float frand( float scale ) {
	g_state = g_state * 1664525u + 1013904223u;
	return ( static_cast<int>( g_state >> 16 ) % 2001 - 1000 ) / 1000.0f * scale;
}

static pod::MemoryArena g_arena{};

static pod::TensorView make( std::vector<float>& host, pod::DType dt, std::initializer_list<int> shape, float scale ) {
	int s[4] = {};
	int nd = 0;
	size_t n = 1;
	for ( int d : shape ) { s[nd++] = d; n *= static_cast<size_t>( d ); }
	pod::TensorView t = core::arena_alloc_tensor( g_arena, dt, s, nd );
	host.resize( n );
	half_t* p = t.ptr<half_t>();
	for ( size_t i = 0; i < n; ++i ) {
		p[i] = backend::from_float_half( frand( scale ) );
		host[i] = backend::to_float( p[i] ); // reference sees the rounded inputs
	}
	return t;
}

static bool check( const char* name, const std::vector<double>& ref, const half_t* got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const double b = backend::to_float( got[i] );
		dot += ref[i] * b; na += ref[i] * ref[i]; nb += b * b;
		maxd = std::fmax( maxd, std::fabs( ref[i] - b ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-28s cos=%.6f maxd=%.5f (max %.3f) %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}

// Runs fn in a child process; returns its exit status and captures stderr
static int run_child( void ( *fn )(), std::string& err ) {
	int fds[2];
	if ( pipe( fds ) != 0 ) return -1;
	const pid_t pid = fork();
	if ( pid == 0 ) {
		dup2( fds[1], 2 );
		close( fds[0] );
		close( fds[1] );
		fn();
		_exit( 0 );
	}
	close( fds[1] );
	char buf[512];
	ssize_t n;
	err.clear();
	while ( ( n = read( fds[0], buf, sizeof( buf ) ) ) > 0 ) err.append( buf, static_cast<size_t>( n ) );
	close( fds[0] );
	int status = 0;
	waitpid( pid, &status, 0 );
	return WIFEXITED( status ) ? WEXITSTATUS( status ) : -1;
}

static void call_unsupported() {
	( void ) engine::modules::ExpertTier::resident_fraction();
}

int main() {
	std::printf( "=== CPU backend smoke (module API -> Slang host kernels) ===\n" );
	bool pass = true;
	ENGINE_ASSERT( core::arena_create( g_arena, 0, 64ull << 20 ), "cpu arena" );
	pod::BatchRows* rows_dev = static_cast<pod::BatchRows*>( core::arena_alloc( g_arena, sizeof( pod::BatchRows ) ) );

	{ // RMSNorm
		const int rows = 4, hs = 1000;
		std::vector<float> x, w;
		auto tx = make( x, pod::DType::FP16, { rows, hs }, 1.0f );
		auto tw = make( w, pod::DType::FP16, { hs }, 0.5f );
		int sh[2] = { rows, hs };
		auto to = core::arena_alloc_tensor( g_arena, pod::DType::FP16, sh, 2 );
		modules::RMSNorm::forward( tx, tw, to, 1e-6f, 1.0f );
		std::vector<double> ref( rows * hs );
		for ( int r = 0; r < rows; ++r ) {
			double sq = 0; for ( int i = 0; i < hs; ++i ) sq += double( x[r * hs + i] ) * x[r * hs + i];
			const double inv = 1.0 / std::sqrt( sq / hs + 1e-6 );
			for ( int i = 0; i < hs; ++i ) ref[r * hs + i] = x[r * hs + i] * ( w[i] + 1.0 ) * inv;
		}
		pass = check( "RMSNorm::forward", ref, to.ptr<half_t>() ) && pass;
	}
	{ // SwiGLU (separate and fused)
		const int rows = 3, inter = 700;
		std::vector<float> g, u, gu;
		auto tg = make( g, pod::DType::FP16, { rows, inter }, 4.0f );
		auto tu = make( u, pod::DType::FP16, { rows, inter }, 4.0f );
		int sh[2] = { rows, inter };
		auto to = core::arena_alloc_tensor( g_arena, pod::DType::FP16, sh, 2 );
		modules::SwiGLU::forward( tg, tu, to );
		std::vector<double> ref( rows * inter );
		for ( int i = 0; i < rows * inter; ++i ) ref[i] = g[i] / ( 1.0 + std::exp( -double( g[i] ) ) ) * u[i];
		pass = check( "SwiGLU::forward", ref, to.ptr<half_t>() ) && pass;

		auto tgu = make( gu, pod::DType::FP16, { rows, 2 * inter }, 4.0f );
		auto to2 = core::arena_alloc_tensor( g_arena, pod::DType::FP16, sh, 2 );
		modules::SwiGLU::forward_fused( tgu, to2 );
		for ( int m = 0; m < rows; ++m ) for ( int c = 0; c < inter; ++c ) {
			const double gv = gu[m * 2 * inter + c], uv = gu[m * 2 * inter + inter + c];
			ref[m * inter + c] = gv / ( 1.0 + std::exp( -gv ) ) * uv;
		}
		pass = check( "SwiGLU::forward_fused", ref, to2.ptr<half_t>() ) && pass;
	}
	{ // RoPE, both pairings
		const int rows = 3, qh = 4, kh = 2, hd = 64, rot = 32;
		const float base = 1000000.0f;
		for ( int il = 0; il < 2; ++il ) {
			std::vector<float> q, k;
			auto tq = make( q, pod::DType::FP16, { rows, qh * hd }, 1.0f );
			auto tk = make( k, pod::DType::FP16, { rows, kh * hd }, 1.0f );
			*rows_dev = pod::BatchRows::causal( rows, 1000, 0 );
			modules::RoPE::forward( tq, tk, qh, kh, hd, rot, rows_dev, rows, base, nullptr, il != 0 );
			std::vector<double> ref( q.size() );
			for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < qh; ++h ) for ( int c = 0; c < hd; ++c ) {
				const size_t o = ( static_cast<size_t>( r ) * qh + h ) * hd;
				double v = q[o + c];
				if ( c < rot ) {
					const int i = il ? c >> 1 : ( c < rot / 2 ? c : c - rot / 2 );
					const double th = ( 1000 + r ) * std::pow( double( base ), -2.0 * i / rot );
					const int p0 = il ? 2 * i : i, p1 = il ? 2 * i + 1 : i + rot / 2;
					const double a = q[o + p0], b = q[o + p1];
					v = ( c == p0 ) ? a * std::cos( th ) - b * std::sin( th ) : a * std::sin( th ) + b * std::cos( th );
				}
				ref[o + c] = v;
			}
			pass = check( il ? "RoPE::forward interleaved" : "RoPE::forward contiguous", ref, tq.ptr<half_t>() ) && pass;
		}
	}
	{ // Elementwise::add_residual, argmax
		const int n = 3000;
		std::vector<float> h, r;
		auto th = make( h, pod::DType::FP16, { 1, n }, 1.0f );
		auto tr = make( r, pod::DType::FP16, { 1, n }, 1.0f );
		modules::Elementwise::add_residual( th, tr );
		std::vector<double> ref( n );
		for ( int i = 0; i < n; ++i ) ref[i] = double( h[i] ) + r[i];
		pass = check( "Elementwise::add_residual", ref, th.ptr<half_t>() ) && pass;

		int* idx = static_cast<int*>( core::arena_alloc( g_arena, sizeof( int ) ) );
		modules::Elementwise::argmax( th, idx, n );
		int best = 0;
		for ( int i = 1; i < n; ++i ) if ( backend::to_float( th.ptr<half_t>()[i] ) > backend::to_float( th.ptr<half_t>()[best] ) ) best = i;
		const bool ok = *idx == best;
		std::printf( "  %-28s idx=%d expect=%d %s\n", "Elementwise::argmax", *idx, best, ok ? "PASS" : "FAIL" );
		pass = ok && pass;
	}
	{ // Embedding (BF16 table -> FP16)
		const int vocab = 16, hs = 300, count = 3;
		uint16_t* table = static_cast<uint16_t*>( core::arena_alloc( g_arena, sizeof( uint16_t ) * vocab * hs ) );
		std::vector<double> tf( vocab * hs );
		for ( int i = 0; i < vocab * hs; ++i ) {
			const backend::bf16_t b = backend::from_float_bf16( frand( 2.0f ) );
			table[i] = b.bits;
			tf[i] = backend::to_float( b );
		}
		int* ids = static_cast<int*>( core::arena_alloc( g_arena, sizeof( int ) * count ) );
		ids[0] = 3; ids[1] = 15; ids[2] = 0;
		int ts[2] = { vocab, hs };
		pod::TensorView tt = pod::make_tensor_2d( table, pod::DType::BF16, ts[0], ts[1] );
		int os[2] = { count, hs };
		auto to = core::arena_alloc_tensor( g_arena, pod::DType::FP16, os, 2 );
		modules::Embedding::forward( tt, ids, count, to );
		std::vector<double> ref( count * hs );
		for ( int r = 0; r < count; ++r ) for ( int c = 0; c < hs; ++c ) ref[r * hs + c] = tf[ids[r] * hs + c];
		pass = check( "Embedding::forward", ref, to.ptr<half_t>() ) && pass;
	}
	{ // GDN::l2_norm_heads and gated_rmsnorm
		const int rows = 2, kh = 2, hd = 64, stride = 4 * kh * hd;
		std::vector<float> qkv, z, gw;
		auto tq = make( qkv, pod::DType::FP16, { rows, stride }, 1.0f );
		modules::GDN::l2_norm_heads( tq, kh, hd, nullptr, false, rows, stride );
		std::vector<double> ref( qkv.size() );
		for ( int r = 0; r < rows; ++r ) for ( int b = 0; b < 2 * kh; ++b ) {
			double sq = 0; for ( int i = 0; i < hd; ++i ) { const double v = qkv[r * stride + b * hd + i]; sq += v * v; }
			const double inv = 1.0 / std::sqrt( sq + 1e-6 );
			for ( int i = 0; i < hd; ++i ) ref[r * stride + b * hd + i] = qkv[r * stride + b * hd + i] * inv;
		}
		for ( int r = 0; r < rows; ++r ) for ( int i = 2 * kh * hd; i < stride; ++i ) ref[r * stride + i] = qkv[r * stride + i];
		pass = check( "GDN::l2_norm_heads", ref, tq.ptr<half_t>() ) && pass;

		const int vh = 3;
		std::vector<float> o;
		auto to = make( o, pod::DType::FP16, { rows, vh * hd }, 1.0f );
		auto tz = make( z, pod::DType::FP16, { rows, vh * hd }, 1.0f );
		auto tw = make( gw, pod::DType::FP16, { hd }, 1.0f );
		modules::GDN::gated_rmsnorm( to, tz, tw, vh, hd, nullptr, false, rows );
		std::vector<double> ref2( o.size() );
		for ( int g = 0; g < rows * vh; ++g ) {
			double sq = 0; for ( int i = 0; i < hd; ++i ) sq += double( o[g * hd + i] ) * o[g * hd + i];
			const double inv = 1.0 / std::sqrt( sq / hd + 1e-6 );
			for ( int i = 0; i < hd; ++i ) {
				const double zv = z[g * hd + i];
				ref2[g * hd + i] = o[g * hd + i] * inv * gw[i] * ( zv / ( 1.0 + std::exp( -zv ) ) );
			}
		}
		pass = check( "GDN::gated_rmsnorm", ref2, to.ptr<half_t>() ) && pass;
	}
	{ // LinearPlainGEMV (FP16 path of Linear::forward)
		const int m = 3, n = 70, k = 130;
		std::vector<float> x, w;
		auto tx = make( x, pod::DType::FP16, { m, k }, 1.0f );
		auto tw = make( w, pod::DType::FP16, { k, n }, 0.3f ); // memory is [n, k] (shape tagged [k, n]), as the loader stores it
		int sh[2] = { m, n };
		auto to = core::arena_alloc_tensor( g_arena, pod::DType::FP16, sh, 2 );
		modules::LinearPlainGEMV::forward( tw.ptr<half_t>(), tx.ptr<half_t>(), to.ptr<half_t>(), m, n, k, nullptr );
		std::vector<double> ref( m * n );
		for ( int r = 0; r < m; ++r ) for ( int c = 0; c < n; ++c ) {
			double a = 0; for ( int i = 0; i < k; ++i ) a += double( x[r * k + i] ) * w[c * k + i];
			ref[r * n + c] = a;
		}
		pass = check( "LinearPlainGEMV::forward", ref, to.ptr<half_t>() ) && pass;
	}
	{ // HyperHead::mean (mean over the 4 streams)
		const int rows = 2, D = 300;
		float* streams = static_cast<float*>( core::arena_alloc( g_arena, sizeof( float ) * rows * 4 * D ) );
		for ( int i = 0; i < rows * 4 * D; ++i ) streams[i] = frand( 1.0f );
		pod::TensorView ts = pod::make_tensor_2d( streams, pod::DType::FP32, rows, 4 * D );
		int sh[2] = { rows, D };
		auto to = core::arena_alloc_tensor( g_arena, pod::DType::FP16, sh, 2 );
		modules::HyperHead::mean( ts, to, D, nullptr, rows );
		std::vector<double> ref( rows * D );
		for ( int r = 0; r < rows; ++r ) for ( int d = 0; d < D; ++d ) {
			double a = 0; for ( int h = 0; h < 4; ++h ) a += streams[r * 4 * D + h * D + d];
			ref[r * D + d] = a / 4.0;
		}
		pass = check( "HyperHead::mean", ref, to.ptr<half_t>() ) && pass;
	}
	{ // GraphCache runs the body directly when the backend has no graphs
		core::GraphCache gc;
		int calls = 0;
		for ( int i = 0; i < 4; ++i ) gc.run( 1, nullptr, [&]( backend::stream_t ) { ++calls; } );
		const bool ok = calls == 4 && gc.replays() == 0 && ENGINE_BACKEND_HAS_GRAPHS == 0;
		std::printf( "  %-28s body runs=%d %s\n", "GraphCache (eager)", calls, ok ? "PASS" : "FAIL" );
		pass = ok && pass;
	}
	{ // A CUDA/SM70-only entry point aborts with the named message
		std::string err;
		const int code = run_child( call_unsupported, err );
		const bool ok = code == 1 && err.find( "CPU backend: ExpertTier::resident_fraction is not implemented" ) != std::string::npos;
		std::printf( "  %-28s exit=%d %s\n", "ExpertTier stub", code, ok ? "PASS" : "FAIL" );
		pass = ok && pass;
	}

	core::arena_destroy( g_arena );
	std::printf( "%s\n", pass ? "ALL PASS" : "FAILED" );
	return pass ? 0 : 1;
}
