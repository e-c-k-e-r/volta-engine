#include "engine/core/arena.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/rope.h"
#include "engine/modules/embedding.h"
#include "engine/backend/runtime.h"
#include "shaders_cuda.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

// Tier 2 parity of the Slang-generated CUDA kernels (src/shaders/*.slang) against the engine's hand-written kernels.
// The Slang versions use a different reduction order, so the check is the tier 2 tolerance, not bit equality.

using namespace engine;

static uint32_t g_state = 12345u;
static float frand( float scale ) {
	g_state = g_state * 1664525u + 1013904223u;
	return ( static_cast<int>( g_state >> 16 ) % 2001 - 1000 ) / 1000.0f * scale;
}

// Graph-timed microseconds per launch (64 launches captured, 20 replays)
template <typename F>
static double time_us( backend::stream_t s, F&& launch ) {
	const int L = 64, reps = 20;
	backend::graph_capture_begin( s );
	for ( int i = 0; i < L; ++i ) launch();
	backend::graph_t g = backend::graph_capture_end( s );
	backend::graph_exec_t e = backend::graph_instantiate( g );
	for ( int i = 0; i < 3; ++i ) backend::graph_launch( e, s );
	backend::stream_sync( s );
	cudaEvent_t e0, e1;
	cudaEventCreate( &e0 ); cudaEventCreate( &e1 );
	cudaEventRecord( e0, s );
	for ( int i = 0; i < reps; ++i ) backend::graph_launch( e, s );
	cudaEventRecord( e1, s );
	cudaEventSynchronize( e1 );
	float ms = 0;
	cudaEventElapsedTime( &ms, e0, e1 );
	backend::graph_destroy( g );
	backend::graph_exec_destroy( e );
	return ms * 1000.0 / reps / L;
}

struct Buf {
	half* d = nullptr;
	size_t n = 0;
	explicit Buf( size_t count, float scale = 1.0f ) : n( count ) {
		d = static_cast<half*>( backend::device_malloc( count * sizeof( half ) ) );
		std::vector<half> h( count );
		for ( auto& v : h ) v = __float2half( frand( scale ) );
		backend::memcpy_sync( d, h.data(), count * sizeof( half ), backend::Copy::H2D );
	}
	~Buf() { backend::device_free( d ); }
	std::vector<half> get() const {
		std::vector<half> h( n );
		backend::memcpy_sync( h.data(), d, n * sizeof( half ), backend::Copy::D2H );
		return h;
	}
	void set( const std::vector<half>& h ) { backend::memcpy_sync( d, h.data(), n * sizeof( half ), backend::Copy::H2D ); }
};

static bool compare( const char* name, const std::vector<half>& ref, const std::vector<half>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	size_t exact = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const double a = __half2float( ref[i] ), b = __half2float( got[i] );
		dot += a * b; na += a * a; nb += b * b;
		maxd = std::fmax( maxd, std::fabs( a - b ) );
		maxabs = std::fmax( maxabs, std::fabs( a ) );
		exact += __half_as_ushort( ref[i] ) == __half_as_ushort( got[i] );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-22s cos %.7f max|d| %.3g (max %.3g) bit-equal %5.1f%%  %s\n", name, cosv, maxd, maxabs,
		100.0 * exact / ref.size(), ok ? "PASS" : "FAIL" );
	return ok;
}

static bool compare_ref( const char* name, const std::vector<double>& ref, const std::vector<half>& got ) {
	std::vector<half> r( ref.size() );
	for ( size_t i = 0; i < ref.size(); ++i ) r[i] = __float2half( static_cast<float>( ref[i] ) );
	return compare( name, r, got );
}

int main() {
	backend::set_device( 0 );
	backend::stream_t s = backend::stream_create();
	bool pass = true;
	std::printf( "=== Slang kernels vs engine kernels (CUDA) ===\n" );

	{ // rms_norm, with and without weight
		const int rows = 5, hidden = 5120;
		Buf x( rows * hidden ), w( hidden, 0.5f ), ref_out( rows * hidden ), got_out( rows * hidden );
		pod::TensorView tx = pod::make_tensor_2d( x.d, pod::DType::FP16, rows, hidden ), tw = pod::make_tensor_1d( w.d, pod::DType::FP16, hidden );
		pod::TensorView tr = pod::make_tensor_2d( ref_out.d, pod::DType::FP16, rows, hidden );
		for ( float bias : { 0.0f, 1.0f } ) {
			modules::RMSNorm::forward( tx, tw, tr, 1e-6f, bias, s );
			slang_rms_norm( x.d, got_out.d, w.d, rows, hidden, 1e-6f, bias, s );
			backend::stream_sync( s );
			pass = compare( bias == 0.0f ? "rms_norm bias 0" : "rms_norm bias 1", ref_out.get(), got_out.get() ) && pass;
		}
	}
	{ // rms_norm_heads (in place)
		const int rows = 3, heads = 24, hd = 256;
		Buf a( rows * heads * hd ), b( rows * heads * hd ), w( hd, 0.5f );
		b.set( a.get() );
		pod::TensorView ta = pod::make_tensor_2d( a.d, pod::DType::FP16, rows, heads * hd ), tw = pod::make_tensor_1d( w.d, pod::DType::FP16, hd );
		modules::RMSNorm::forward_heads( ta, tw, heads, hd, 1e-6f, 1.0f, s );
		slang_rms_norm_heads( b.d, w.d, rows, heads, hd, 1e-6f, 1.0f, s );
		backend::stream_sync( s );
		pass = compare( "rms_norm_heads", a.get(), b.get() ) && pass;
	}
	{ // add_residual
		const int n = 5 * 5120;
		Buf h1( n ), r( n ), h2( n );
		h2.set( h1.get() );
		pod::TensorView t1 = pod::make_tensor_2d( h1.d, pod::DType::FP16, 5, 5120 ), tr = pod::make_tensor_2d( r.d, pod::DType::FP16, 5, 5120 );
		modules::Elementwise::add_residual( t1, tr, s );
		slang_add_residual( h2.d, r.d, n, s );
		backend::stream_sync( s );
		pass = compare( "add_residual", h1.get(), h2.get() ) && pass;
	}
	{ // swiglu, with and without a clamp limit
		const int n = 7 * 4096;
		Buf g( n, 4.0f ), u( n, 4.0f ), o1( n ), o2( n );
		pod::TensorView tg = pod::make_tensor_2d( g.d, pod::DType::FP16, 7, 4096 ), tu = pod::make_tensor_2d( u.d, pod::DType::FP16, 7, 4096 );
		pod::TensorView to = pod::make_tensor_2d( o1.d, pod::DType::FP16, 7, 4096 );
		for ( float limit : { 0.0f, 3.0f } ) {
			modules::SwiGLU::forward( tg, tu, to, s, limit );
			slang_swiglu( g.d, u.d, o2.d, n, limit, s );
			backend::stream_sync( s );
			pass = compare( limit == 0.0f ? "swiglu" : "swiglu limit 3", o1.get(), o2.get() ) && pass;
		}
	}
	{ // swiglu_fused
		const int rows = 6, inter = 2048;
		Buf gu( rows * 2 * inter, 4.0f ), o1( rows * inter ), o2( rows * inter );
		pod::TensorView tgu = pod::make_tensor_2d( gu.d, pod::DType::FP16, rows, 2 * inter ), to = pod::make_tensor_2d( o1.d, pod::DType::FP16, rows, inter );
		modules::SwiGLU::forward_fused( tgu, to, s, 0.0f );
		slang_swiglu_fused( gu.d, o2.d, inter, rows * inter, 0.0f, s );
		backend::stream_sync( s );
		pass = compare( "swiglu_fused", o1.get(), o2.get() ) && pass;
	}
	{ // embed_tokens from a BF16 table
		const int vocab = 512, hidden = 4096, count = 7;
		std::vector<uint16_t> table( static_cast<size_t>( vocab ) * hidden );
		for ( auto& v : table ) { const float f = frand( 2.0f ); uint32_t bits; std::memcpy( &bits, &f, 4 ); v = static_cast<uint16_t>( bits >> 16 ); }
		void* d_table = backend::device_malloc( table.size() * 2 );
		backend::memcpy_sync( d_table, table.data(), table.size() * 2, backend::Copy::H2D );
		int ids[count] = { 3, 511, 0, 77, 200, 5, 400 };
		void* d_ids = backend::device_malloc( sizeof( ids ) );
		backend::memcpy_sync( d_ids, ids, sizeof( ids ), backend::Copy::H2D );
		Buf o1( count * hidden ), o2( count * hidden );
		pod::TensorView tt = pod::make_tensor_2d( d_table, pod::DType::BF16, vocab, hidden ), to = pod::make_tensor_2d( o1.d, pod::DType::FP16, count, hidden );
		modules::Embedding::forward( tt, static_cast<const int*>( d_ids ), count, to, s );
		slang_embed_tokens( d_table, d_ids, o2.d, hidden, count, s );
		backend::stream_sync( s );
		pass = compare( "embed_tokens", o1.get(), o2.get() ) && pass;
		backend::device_free( d_table );
		backend::device_free( d_ids );
	}
	{ // rope, both pairings
		const int rows = 6, qh = 8, kh = 2, hd = 128, rot = 64;
		pod::BatchRows br = pod::BatchRows::causal( rows, 1000, 0 );
		void* d_rows = backend::device_malloc( sizeof( br ) );
		backend::memcpy_sync( d_rows, &br, sizeof( br ), backend::Copy::H2D );
		int* d_pos = reinterpret_cast<int*>( d_rows ) + ( reinterpret_cast<int*>( &br.pos ) - reinterpret_cast<int*>( &br ) );
		for ( bool il : { false, true } ) {
			Buf q1( rows * qh * hd ), k1( rows * kh * hd ), q2( rows * qh * hd ), k2( rows * kh * hd );
			q2.set( q1.get() ); k2.set( k1.get() );
			pod::TensorView tq = pod::make_tensor_2d( q1.d, pod::DType::FP16, rows, qh * hd ), tk = pod::make_tensor_2d( k1.d, pod::DType::FP16, rows, kh * hd );
			modules::RoPE::forward( tq, tk, qh, kh, hd, rot, static_cast<const pod::BatchRows*>( d_rows ), rows, 1000000.0f, s, il );
			slang_rope( q2.d, k2.d, d_pos, rows, qh, kh, hd, rot, il ? 1 : 0, 1000000.0f, s );
			backend::stream_sync( s );
			pass = compare( il ? "rope interleaved q" : "rope contiguous q", q1.get(), q2.get() ) && pass;
			pass = compare( il ? "rope interleaved k" : "rope contiguous k", k1.get(), k2.get() ) && pass;
		}
		backend::device_free( d_rows );
	}

	{ // moe_sum_planes and moe_accumulate against CPU references
		const int n = 4096, groups = 5;
		Buf planes( n * groups ), out( n );
		slang_moe_sum_planes( planes.d, out.d, n, groups, s );
		backend::stream_sync( s );
		auto hp = planes.get();
		std::vector<double> ref( n );
		for ( int i = 0; i < n; ++i ) { float a = __half2float( hp[i] ); for ( int g = 1; g < groups; ++g ) a += __half2float( hp[g * n + i] ); ref[i] = a; }
		pass = compare_ref( "moe_sum_planes", ref, out.get() ) && pass;
		Buf acc( n ), eo( n );
		auto ha = acc.get(); auto he = eo.get();
		slang_moe_accumulate( acc.d, eo.d, 0.37f, n, s );
		backend::stream_sync( s );
		for ( int i = 0; i < n; ++i ) ref[i] = std::fma( static_cast<double>( __half2float( he[i] ) ), 0.37f, static_cast<double>( __half2float( ha[i] ) ) );
		pass = compare_ref( "moe_accumulate", ref, acc.get() ) && pass;
	}
	{ // batched_argmax: lowest index among ties, offset applied
		const int batch = 4, numel = 7000, stride = 7168, offset = 100;
		Buf logits( batch * stride, 3.0f );
		int* d_idx = static_cast<int*>( backend::device_malloc( batch * sizeof( int ) ) );
		float* d_val = static_cast<float*>( backend::device_malloc( batch * sizeof( float ) ) );
		slang_batched_argmax( logits.d, d_idx, d_val, numel, batch, offset, stride, s );
		backend::stream_sync( s );
		int idx[batch]; float val[batch];
		backend::memcpy_sync( idx, d_idx, sizeof( idx ), backend::Copy::D2H );
		backend::memcpy_sync( val, d_val, sizeof( val ), backend::Copy::D2H );
		auto hl = logits.get();
		bool ok = true;
		for ( int r = 0; r < batch; ++r ) {
			int bi = -1; float bv = -1e30f;
			for ( int i = 0; i < numel; ++i ) { const float v = __half2float( hl[r * stride + i] ); if ( v > bv ) { bv = v; bi = i; } }
			ok = ok && idx[r] == bi + offset && val[r] == bv;
		}
		std::printf( "  %-22s indices+values vs CPU (ties to lowest index)  %s\n", "batched_argmax", ok ? "PASS" : "FAIL" );
		pass = ok && pass;
		backend::device_free( d_idx ); backend::device_free( d_val );
	}
	{ // router_select vs a CPU reference
		const int rows = 3, E = 288, K = 8, slices = 32, maxs = 32;
		std::vector<float> partials( static_cast<size_t>( rows ) * maxs * 288 );
		for ( auto& v : partials ) v = frand( 0.25f );
		Buf bias( E, 0.1f );
		float* d_part = static_cast<float*>( backend::device_malloc( partials.size() * 4 ) );
		backend::memcpy_sync( d_part, partials.data(), partials.size() * 4, backend::Copy::H2D );
		int* d_idx = static_cast<int*>( backend::device_malloc( rows * K * sizeof( int ) ) );
		Buf w( rows * K );
		slang_router_select( d_part, bias.d, d_idx, w.d, rows, E, K, slices, maxs, 2.5f, s );
		backend::stream_sync( s );
		std::vector<int> idx( rows * K );
		backend::memcpy_sync( idx.data(), d_idx, idx.size() * 4, backend::Copy::D2H );
		auto hb = bias.get();
		std::vector<double> ref( rows * K );
		bool ok = true;
		for ( int r = 0; r < rows; ++r ) {
			std::vector<float> raw( E ), sel( E );
			for ( int e = 0; e < E; ++e ) {
				float logit = 0; for ( int ks = 0; ks < slices; ++ks ) logit += partials[( static_cast<size_t>( r ) * maxs + ks ) * 288 + e];
				raw[e] = 1.0f / ( 1.0f + std::exp( -logit ) );
				sel[e] = raw[e] + __half2float( hb[e] );
			}
			std::vector<int> chosen; double total = 0;
			for ( int k = 0; k < K; ++k ) {
				int bi = -1; for ( int e = 0; e < E; ++e ) { bool taken = false; for ( int c : chosen ) taken = taken || c == e; if ( !taken && ( bi < 0 || sel[e] > sel[bi] ) ) bi = e; }
				chosen.push_back( bi ); total += raw[bi];
			}
			for ( int k = 0; k < K; ++k ) { ok = ok && idx[r * K + k] == chosen[k]; ref[r * K + k] = raw[chosen[k]] * 2.5 / ( total + 1e-9 ); }
		}
		std::printf( "  %-22s expert ids vs CPU  %s\n", "router_select", ok ? "PASS" : "FAIL" );
		pass = ok && compare_ref( "router_select weights", ref, w.get() ) && pass;
		backend::device_free( d_part ); backend::device_free( d_idx );
	}
	{ // speed, for reference: the CUDA build keeps its own kernels unless a Slang one matches
		const int rows = 1, hidden = 5120;
		Buf x( rows * hidden ), w( hidden, 0.5f ), o( rows * hidden ), g( 4096 * 8, 2.0f ), u( 4096 * 8, 2.0f ), so( 4096 * 8 );
		pod::TensorView tx = pod::make_tensor_2d( x.d, pod::DType::FP16, rows, hidden ), tw = pod::make_tensor_1d( w.d, pod::DType::FP16, hidden );
		pod::TensorView to = pod::make_tensor_2d( o.d, pod::DType::FP16, rows, hidden );
		pod::TensorView tg = pod::make_tensor_2d( g.d, pod::DType::FP16, 8, 4096 ), tu = pod::make_tensor_2d( u.d, pod::DType::FP16, 8, 4096 );
		pod::TensorView tso = pod::make_tensor_2d( so.d, pod::DType::FP16, 8, 4096 );
		std::printf( "--- graph-timed, us per launch (engine / slang) ---\n" );
		std::printf( "  rms_norm 1x5120      %6.2f / %6.2f\n", time_us( s, [&] { modules::RMSNorm::forward( tx, tw, to, 1e-6f, 0.0f, s ); } ),
			time_us( s, [&] { slang_rms_norm( x.d, o.d, w.d, rows, hidden, 1e-6f, 0.0f, s ); } ) );
		std::printf( "  swiglu 8x4096        %6.2f / %6.2f\n", time_us( s, [&] { modules::SwiGLU::forward( tg, tu, tso, s, 0.0f ); } ),
			time_us( s, [&] { slang_swiglu( g.d, u.d, so.d, 8 * 4096, 0.0f, s ); } ) );
	}

	std::printf( pass ? "[+] Slang kernel parity PASS\n" : "[-] Slang kernel parity FAIL\n" );
	return pass ? 0 : 1;
}
