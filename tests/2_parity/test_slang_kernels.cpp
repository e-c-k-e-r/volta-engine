#include "engine/core/arena.h"
#include "engine/modules/norm.h"
#include "engine/modules/swiglu.h"
#include "engine/modules/rope.h"
#include "engine/modules/embedding.h"
#include "engine/modules/gdn.h"
#include "engine/modules/mhc.h"
#include "engine/modules/dsa.h"
#include "engine/modules/attention.h"
#include "engine/spec/tap_ring.h"
#include "engine/backend/runtime.h"
#include "shaders_cuda.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

// Tier 2 parity of the Slang-generated CUDA kernels (src/kernels/slang/*.slang) against the engine's hand-written kernels.
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

// FP32 / INT32 device buffer (the FP16 Buf above rounds its contents)
template <typename T>
struct DBuf {
	T* d = nullptr;
	size_t n = 0;
	explicit DBuf( size_t count ) : n( count ) {
		d = static_cast<T*>( backend::device_malloc( count * sizeof( T ) ) );
		backend::memset_sync( d, 0, count * sizeof( T ) );
	}
	~DBuf() { backend::device_free( d ); }
	void set( const std::vector<T>& h ) { backend::memcpy_sync( d, h.data(), n * sizeof( T ), backend::Copy::H2D ); }
	std::vector<T> get() const {
		std::vector<T> h( n );
		backend::memcpy_sync( h.data(), d, n * sizeof( T ), backend::Copy::D2H );
		return h;
	}
};
static std::vector<float> rnd( size_t n, float scale ) {
	std::vector<float> v( n );
	for ( auto& x : v ) x = frand( scale );
	return v;
}
static bool compare_f( const char* name, const std::vector<float>& ref, const std::vector<float>& got ) {
	double dot = 0, na = 0, nb = 0, maxd = 0, maxabs = 0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		dot += static_cast<double>( ref[i] ) * got[i]; na += static_cast<double>( ref[i] ) * ref[i]; nb += static_cast<double>( got[i] ) * got[i];
		maxd = std::fmax( maxd, std::fabs( static_cast<double>( ref[i] ) - got[i] ) );
		maxabs = std::fmax( maxabs, std::fabs( ref[i] ) );
	}
	const double cosv = dot / ( std::sqrt( na ) * std::sqrt( nb ) + 1e-30 );
	const bool ok = cosv >= 0.9999 && maxd <= 0.005 * maxabs + 1e-3;
	std::printf( "  %-22s cos %.7f max|d| %.3g (max %.3g)                    %s\n", name, cosv, maxd, maxabs, ok ? "PASS" : "FAIL" );
	return ok;
}
// Device copy of a BatchRows; slot / pos arrays are addressed inside it
struct DRows {
	pod::BatchRows* d = nullptr;
	explicit DRows( const pod::BatchRows& r ) {
		d = static_cast<pod::BatchRows*>( backend::device_malloc( sizeof( pod::BatchRows ) ) );
		backend::memcpy_sync( d, &r, sizeof( r ), backend::Copy::H2D );
	}
	~DRows() { backend::device_free( d ); }
	void* slots() const { return reinterpret_cast<char*>( d ) + offsetof( pod::BatchRows, slot ); }
	void* pos() const { return reinterpret_cast<char*>( d ) + offsetof( pod::BatchRows, pos ); }
};

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
	{ // l2_norm_heads (q and k slices of a packed row, q scaled) and gated_rmsnorm, against the engine kernels
		const int rows = 3, kh = 4, hd = 128, stride = 2 * kh * hd + 512;
		Buf a( rows * stride ), b( rows * stride );
		b.set( a.get() );
		pod::TensorView ta = pod::make_tensor_2d( a.d, pod::DType::FP16, rows, stride );
		modules::GDN::l2_norm_heads( ta, kh, hd, s, true, rows, stride );
		slang_l2_norm_heads( b.d, rows, kh, hd, 1, stride, s );
		backend::stream_sync( s );
		pass = compare( "l2_norm_heads", a.get(), b.get() ) && pass;
		const int vh = 8;
		for ( bool sig : { false, true } ) {
			Buf o1( rows * vh * hd ), o2( rows * vh * hd ), z( rows * vh * hd, 2.0f ), gw( hd, 0.5f );
			o2.set( o1.get() );
			pod::TensorView to = pod::make_tensor_2d( o1.d, pod::DType::FP16, rows, vh * hd ), tz = pod::make_tensor_2d( z.d, pod::DType::FP16, rows, vh * hd );
			pod::TensorView tgw = pod::make_tensor_1d( gw.d, pod::DType::FP16, hd );
			modules::GDN::gated_rmsnorm( to, tz, tgw, vh, hd, s, sig, rows );
			slang_gated_rmsnorm( o2.d, z.d, gw.d, rows, vh, hd, sig ? 1 : 0, s );
			backend::stream_sync( s );
			pass = compare( sig ? "gated_rmsnorm sigmoid" : "gated_rmsnorm silu", o1.get(), o2.get() ) && pass;
		}
	}
	{ // GDN gate projection
		const int rows = 3, vh = 32, hid = 2048;
		Buf x( rows * hid ), wa( vh * hid, 0.05f ), wb( vh * hid, 0.05f ), alog( vh, 0.5f ), dt( vh, 0.5f );
		DBuf<float> g1( rows * vh ), b1( rows * vh ), g2( rows * vh ), b2( rows * vh );
		pod::TensorView tx = pod::make_tensor_2d( x.d, pod::DType::FP16, rows, hid ), ta = pod::make_tensor_2d( wa.d, pod::DType::FP16, vh, hid ), tb = pod::make_tensor_2d( wb.d, pod::DType::FP16, vh, hid );
		pod::TensorView tl = pod::make_tensor_1d( alog.d, pod::DType::FP16, vh ), td = pod::make_tensor_1d( dt.d, pod::DType::FP16, vh );
		pod::TensorView tg = pod::make_tensor_2d( g1.d, pod::DType::FP32, rows, vh ), tbe = pod::make_tensor_2d( b1.d, pod::DType::FP32, rows, vh );
		modules::GDN::project_gates( tx, ta, tb, tl, td, tg, tbe, vh, hid, s, rows );
		slang_project_gates( x.d, wa.d, wb.d, alog.d, dt.d, g2.d, b2.d, rows, vh, hid, s );
		backend::stream_sync( s );
		pass = compare_f( "project_gates gate", g1.get(), g2.get() ) && pass;
		pass = compare_f( "project_gates beta", b1.get(), b2.get() ) && pass;
	}
	{ // GDN multi-row conv, tiled prefill conv and history update, multi-row recurrence (two slots, snapshots)
		const int ch = 1000, rows = 5, nslot = 2, snap_rows = 4;
		const size_t ss = ch * 3 + 16;
		const int slot_ids[5] = { 0, 0, 1, 1, 1 }, positions[5] = { 4, 5, 9, 10, 11 };
		DRows rws( pod::BatchRows::batched( rows, slot_ids, positions ) );
		Buf qa( rows * ch ), qb( rows * ch ), w( ch * 4, 0.5f );
		qb.set( qa.get() );
		const auto h_init = rnd( nslot * ss, 1.0f );
		DBuf<float> ha( nslot * ss ), hb( nslot * ss ), sa( rows * ch * 3 ), sb( rows * ch * 3 );
		DBuf<int> on( 1 );
		ha.set( h_init ); hb.set( h_init );
		on.set( { 1 } );
		pod::TensorView tq = pod::make_tensor_2d( qa.d, pod::DType::FP16, rows, ch ), tw = pod::make_tensor_2d( w.d, pod::DType::FP16, ch, 4 );
		modules::GDN::causal_conv1d_silu_rows( tq, ha.d, tw, ch, rows, rws.d, ss, sa.d, ch * 3, snap_rows, on.d, s );
		slang_causal_conv1d_silu_rows( qb.d, hb.d, w.d, rws.slots(), sb.d, on.d, ch, rows, snap_rows, ss, ch * 3, s );
		backend::stream_sync( s );
		pass = compare( "causal_conv1d rows", qa.get(), qb.get() ) && pass;
		pass = compare_f( "conv rows history", ha.get(), hb.get() ) && pass;
		pass = compare_f( "conv rows snap", sa.get(), sb.get() ) && pass;

		const int trows = 50;
		const int cslot[1] = { 1 }, cpos[1] = { 0 };
		DRows crw( pod::BatchRows::batched( 1, cslot, cpos ) );
		Buf src( trows * ch ), da( trows * ch ), db( trows * ch );
		ha.set( h_init ); hb.set( h_init );
		pod::TensorView ts = pod::make_tensor_2d( src.d, pod::DType::FP16, trows, ch ), tdst = pod::make_tensor_2d( da.d, pod::DType::FP16, trows, ch );
		modules::GDN::causal_conv1d_silu_tiled( ts, tdst, ha.d, tw, ch, trows, crw.d, ss, s );
		slang_causal_conv1d_silu_tiled( src.d, db.d, hb.d, w.d, crw.slots(), ch, trows, 16, ss, s );
		slang_causal_conv1d_history( src.d, hb.d, crw.slots(), ch, trows, ss, s );
		backend::stream_sync( s );
		pass = compare( "causal_conv1d tiled", da.get(), db.get() ) && pass;
		pass = compare_f( "conv tiled history", ha.get(), hb.get() ) && pass;

		const int hk = 4, hv = 8, stride = ( 2 * hk + hv ) * 128;
		const size_t rss = static_cast<size_t>( hv ) * 128 * 128;
		Buf qkv( rows * stride, 0.5f ), oa( rows * hv * 128 ), ob( rows * hv * 128 );
		std::vector<float> gate_h( rows * hv ), beta_h( rows * hv );
		for ( auto& v : gate_h ) v = -std::fabs( frand( 0.5f ) );
		for ( auto& v : beta_h ) v = std::fabs( frand( 1.0f ) );
		DBuf<float> gate( rows * hv ), beta( rows * hv ), st_a( nslot * rss ), st_b( nslot * rss ), sn_a( snap_rows * rss ), sn_b( snap_rows * rss );
		gate.set( gate_h ); beta.set( beta_h );
		const auto st_init = rnd( nslot * rss, 0.1f );
		st_a.set( st_init ); st_b.set( st_init );
		pod::TensorView tqkv = pod::make_tensor_2d( qkv.d, pod::DType::FP16, rows, stride ), tgate = pod::make_tensor_2d( gate.d, pod::DType::FP32, rows, hv );
		pod::TensorView tbeta = pod::make_tensor_2d( beta.d, pod::DType::FP32, rows, hv ), tout = pod::make_tensor_2d( oa.d, pod::DType::FP16, rows, hv * 128 );
		modules::GDN::recurrent_step_rows( st_a.d, tqkv, tgate, tbeta, tout, hk, hv, 128, rows, rws.d, rss, sn_a.d, rss, snap_rows, on.d, s );
		slang_recurrent_step_rows( st_b.d, qkv.d, gate.d, beta.d, ob.d, rws.slots(), sn_b.d, on.d, hk, hv, rows, snap_rows, rss, rss, s );
		backend::stream_sync( s );
		pass = compare( "recurrent_step_rows", oa.get(), ob.get() ) && pass;
		pass = compare_f( "recurrent state", st_a.get(), st_b.get() ) && pass;
		pass = compare_f( "recurrent snap", sn_a.get(), sn_b.get() ) && pass;
	}
	{ // FP16 paged KV store, K and V
		const int count = 5, kvh = 4, hd = 64, total = kvh * hd, max_seq = 32, layer = 2, nslot = 3;
		const size_t ss = static_cast<size_t>( 4 ) * max_seq * total + 8;
		const int slot_ids[5] = { 0, 0, 2, 1, 1 }, positions[5] = { 3, 4, 17, 0, 1 };
		DRows rws( pod::BatchRows::batched( count, slot_ids, positions ) );
		Buf k( count * total ), v( count * total ), k1( nslot * ss ), v1( nslot * ss ), k2( nslot * ss ), v2( nslot * ss );
		k2.set( k1.get() ); v2.set( v1.get() );
		pod::TensorView tk = pod::make_tensor_2d( k.d, pod::DType::FP16, count, total ), tv = pod::make_tensor_2d( v.d, pod::DType::FP16, count, total );
		modules::Attention::store_kv_cache( pod::kv_ref_f16( k1.d, v1.d ), tk, tv, ss, layer, rws.d, count, kvh, hd, max_seq, s );
		slang_kv_store_f16( k2.d, k.d, rws.slots(), rws.pos(), ss, layer, count, total, max_seq, s );
		slang_kv_store_f16( v2.d, v.d, rws.slots(), rws.pos(), ss, layer, count, total, max_seq, s );
		backend::stream_sync( s );
		pass = compare( "kv_store_f16 K", k1.get(), k2.get() ) && pass;
		pass = compare( "kv_store_f16 V", v1.get(), v2.get() ) && pass;
	}
	{ // mHC: init, collapse (partials + finalize), update, hyper-head mean
		const int D = 4096, rows = 3, row_len = 4 * D, chunk_cols = 256, nch = row_len / chunk_cols, chunk_cols_c = 1024, iters = 20;
		Buf in( rows * D );
		DBuf<float> s1( rows * row_len ), s2( rows * row_len );
		modules::HyperConnection::init_streams( in.d, s1.d, D, s, rows );
		slang_init_streams( in.d, s2.d, D, rows, s );
		backend::stream_sync( s );
		pass = compare_f( "init_streams", s1.get(), s2.get() ) && pass;

		const auto sv = rnd( rows * row_len, 1.0f );
		s1.set( sv ); s2.set( sv );
		DBuf<float> fn( 24 * row_len ), base( 24 ), scale( 3 ), partials( rows * nch * 25 );
		fn.set( rnd( 24 * row_len, 0.02f ) ); base.set( rnd( 24, 0.5f ) ); scale.set( { 0.7f, 0.9f, 1.1f } );
		DBuf<float> post1( rows * 4 ), comb1( rows * 16 ), inv1( rows ), log1( rows * 24 ), pre1( rows * 4 );
		DBuf<float> post2( rows * 4 ), comb2( rows * 16 ), inv2( rows ), log2( rows * 24 ), pre2( rows * 4 );
		Buf col1( rows * D ), col2( rows * D );
		modules::HyperConnection::collapse( s1.d, fn.d, base.d, scale.d, col1.d, post1.d, comb1.d, inv1.d, log1.d, pre1.d, D, iters, 1e-6f, s, rows );
		slang_hc_mix_partials( s2.d, fn.d, partials.d, row_len, chunk_cols, rows, s );
		slang_hc_mix_finalize( s2.d, partials.d, base.d, scale.d, post2.d, comb2.d, col2.d, inv2.d, log2.d, pre2.d, D, nch, chunk_cols_c, iters, 1e-5f, 1e-6f, rows, s );
		backend::stream_sync( s );
		pass = compare( "hc_mix collapsed", col1.get(), col2.get() ) && pass;
		pass = compare_f( "hc_mix post", post1.get(), post2.get() ) && pass;
		pass = compare_f( "hc_mix comb (sinkhorn)", comb1.get(), comb2.get() ) && pass;
		pass = compare_f( "hc_mix inv_rms", inv1.get(), inv2.get() ) && pass;
		pass = compare_f( "hc_mix logits", log1.get(), log2.get() ) && pass;
		pass = compare_f( "hc_mix pre", pre1.get(), pre2.get() ) && pass;

		Buf y( rows * D );
		modules::HyperConnection::update( s1.d, y.d, post1.d, comb1.d, D, s, rows );
		slang_hc_apply( s2.d, s2.d, y.d, post1.d, comb1.d, D, rows, s );
		backend::stream_sync( s );
		pass = compare_f( "hc_apply", s1.get(), s2.get() ) && pass;

		Buf m1( rows * D ), m2( rows * D );
		modules::HyperHead::mean( s1.d, m1.d, D, s, rows );
		slang_hyperhead_mean( s1.d, m2.d, D, rows, s );
		backend::stream_sync( s );
		pass = compare( "hyperhead_mean", m1.get(), m2.get() ) && pass;
	}
	{ // DSA indexer prep and pool update
		const int rows = 3, layer = 1, max_seq = 2048;
		const size_t pss = 2 * 1024 * 256, qss = 2 * ( max_seq / 4 ) * 128;
		const int slot_ids[3] = { 0, 1, 0 }, positions[3] = { 5, 1030, 7 }, positions2[3] = { 7, 1027, 6 };
		DRows rws( pod::BatchRows::batched( rows, slot_ids, positions ) ), rws2( pod::BatchRows::batched( rows, slot_ids, positions2 ) );
		Buf proj( rows * 288 ), lw( 128, 1.0f ), lb( 128, 0.5f ), pl1( 2 * pss ), pl2( 2 * pss ), w1( rows * 32 ), w2( rows * 32 );
		pl2.set( pl1.get() );
		modules::DSA::prep( proj.d, lw.d, lb.d, pl1.d, w1.d, layer, max_seq, rows, rws.d, pss, s );
		slang_dsa_prep( proj.d, lw.d, lb.d, pl2.d, w2.d, rws.slots(), rws.pos(), layer, pss, rows, s );
		backend::stream_sync( s );
		pass = compare( "dsa_prep plane", pl1.get(), pl2.get() ) && pass;
		pass = compare( "dsa_prep head weights", w1.get(), w2.get() ) && pass;
		Buf ape( 4 * 128, 0.5f ), po1( 2 * qss ), po2( 2 * qss );
		po1.set( std::vector<half>( 2 * qss, __float2half( 0.0f ) ) ); po2.set( po1.get() );
		modules::DSA::pool_update( pl1.d, po1.d, ape.d, layer, max_seq, rows, rws2.d, pss, qss, s );
		slang_dsa_pool_update( pl1.d, po2.d, ape.d, rws2.slots(), rws2.pos(), layer, max_seq, pss, qss, rows, s );
		backend::stream_sync( s );
		pass = compare( "dsa_pool_update", po1.get(), po2.get() ) && pass;
	}
	{ // speculative tap ring store and stream reduction
		const int m = 4, tap = 3, H = 4096, cap = 16;
		const int slot_ids[4] = { 0, 1, 0, 0 }, positions[4] = { 3, 9, 20, 5 };
		DRows rws( pod::BatchRows::batched( m, slot_ids, positions ) );
		DBuf<int> owner( 1 );
		owner.set( { 0 } );
		Buf hidden( m * H ), t1( cap * 5 * H ), t2( cap * 5 * H );
		t2.set( t1.get() );
		spec::dflash_publish_taps( t1.d, hidden.d, rws.d, owner.d, m, tap, H, cap, s );
		slang_dflash_store_taps( t2.d, hidden.d, rws.slots(), rws.pos(), owner.d, m, tap, H, cap, s );
		backend::stream_sync( s );
		pass = compare( "dflash_store_taps", t1.get(), t2.get() ) && pass;
		DBuf<float> streams( m * 4 * H );
		streams.set( rnd( m * 4 * H, 1.0f ) );
		for ( int mode = 0; mode < 5; ++mode ) {
			Buf r1( m * H ), r2( m * H );
			spec::dflash_reduce_streams( streams.d, r1.d, m, H, mode, s );
			slang_reduce_streams( streams.d, r2.d, m, H, mode, s );
			backend::stream_sync( s );
			pass = compare( mode == 0 ? "reduce_streams mean" : "reduce_streams stream", r1.get(), r2.get() ) && pass;
		}
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
