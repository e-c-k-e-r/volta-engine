// Op-level cross-backend golden test for the attention side of the module API: attention decode / prefill (GQA 24/4 x 256, gate),
// Q/gate unpack, MLA latent (absorb_q, store, attend dense and DSA-selected, expand_out), mHC collapse + update, DSA (project, score,
// select) and the GDN / KDA single-row and row forms. ONE source builds in the CUDA configuration (--dump runs the real CUDA kernels
// and writes golden files) and in the CPU configuration (--check recomputes with the CPU backend and compares); --check also
// passes in the CUDA build against its own dump. Inputs come from GoldenRng fixed seeds, so nothing but outputs is stored.
//   test_ops_attention --dump <dir> | --check <dir>
#include "golden_io.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/core/tensor.h"
#include "engine/modules/attention.h"
#include "engine/modules/mla.h"
#include "engine/modules/mhc.h"
#include "engine/modules/dsa.h"
#include "engine/modules/gdn.h"
#include <algorithm>
#include <vector>

using namespace engine;
using backend::half_t;
using modules::Attention;
using modules::MLALatent;
using modules::DSA;
using modules::GDN;
using modules::HyperConnection;

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

std::vector<half_t> rh( GoldenRng& r, size_t n, float scale ) {
	std::vector<half_t> v( n );
	for ( auto& x : v ) x = backend::from_float_half( r.uniform( scale ) );
	return v;
}

std::vector<float> rf( GoldenRng& r, size_t n, float scale, float offset = 0.0f ) {
	std::vector<float> v( n );
	for ( auto& x : v ) x = offset + r.uniform( scale );
	return v;
}

std::vector<float> dl( const half_t* d, size_t n ) {
	std::vector<half_t> h( n );
	backend::memcpy_sync( h.data(), d, n * sizeof( half_t ), backend::Copy::D2H );
	std::vector<float> f( n );
	for ( size_t i = 0; i < n; ++i ) f[i] = backend::to_float( h[i] );
	return f;
}

std::vector<float> dlf( const float* d, size_t n ) {
	std::vector<float> f( n );
	backend::memcpy_sync( f.data(), d, n * sizeof( float ), backend::Copy::D2H );
	return f;
}

std::vector<float> dli( const int* d, size_t n ) {
	std::vector<int> h( n );
	backend::memcpy_sync( h.data(), d, n * sizeof( int ), backend::Copy::D2H );
	return std::vector<float>( h.begin(), h.end() );
}

pod::BatchRows* up_rows( const pod::BatchRows& br ) {
	pod::BatchRows* d = static_cast<pod::BatchRows*>( backend::device_malloc( sizeof( pod::BatchRows ) ) );
	g_allocs.push_back( d );
	backend::memcpy_sync( d, &br, sizeof( br ), backend::Copy::H2D );
	return d;
}

pod::TensorView tv( void* p, pod::DType t, int rows, int cols ) {
	return pod::make_tensor_2d( p, t, rows, cols );
}

// --dump writes the golden file; --check compares the freshly computed output with it
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

// ---- standard attention: GQA 24/4 x 256 like Qwen ----
constexpr int HQ = 24, HKV = 4, HD = 256, A_SEQ = 128, A_LAYERS = 2, A_LAYER = 1, A_SLOTS = 3;

void test_attention() {
	std::printf( "attention (GQA %d/%d x %d)\n", HQ, HKV, HD );
	GoldenRng rng( 1001 );
	const size_t slot_stride = static_cast<size_t>( A_LAYERS ) * A_SEQ * HKV * HD;
	std::vector<half_t> hk = rh( rng, slot_stride * A_SLOTS, 1.0f ), hv = rh( rng, slot_stride * A_SLOTS, 1.0f );
	half_t* dk = up( hk );
	half_t* dv = up( hv );
	pod::KVRef kv = pod::kv_ref_f16( dk, dv );

	// Decode: rows in distinct slots and positions
	const int slots[4] = { 0, 2, 1, 0 };
	const int pos[4] = { 100, 37, 5, 127 };
	const int R = 4;
	pod::BatchRows br = pod::BatchRows::batched( R, slots, pos );
	pod::BatchRows* drows = up_rows( br );
	std::vector<half_t> hq = rh( rng, static_cast<size_t>( R ) * HQ * HD, 1.0f ), hg = rh( rng, hq.size(), 2.0f );
	half_t* dq = up( hq );
	half_t* dg = up( hg );
	half_t* dout = zeros<half_t>( hq.size() );
	pod::TensorView q = tv( dq, pod::DType::FP16, R, HQ * HD ), out = tv( dout, pod::DType::FP16, R, HQ * HD );
	pod::TensorView gate = tv( dg, pod::DType::FP16, R, HQ * HD ), no_gate{};
	Attention::forward_decode( kv, q, no_gate, out, slot_stride, A_LAYER, drows, R, HQ, HKV, HD, A_SEQ );
	backend::device_sync();
	emit( "attn_decode_nogate", dl( dout, hq.size() ) );
	Attention::forward_decode( kv, q, gate, out, slot_stride, A_LAYER, drows, R, HQ, HKV, HD, A_SEQ );
	backend::device_sync();
	emit( "attn_decode_gate", dl( dout, hq.size() ) );

	// Prefill chunk: 40 causal rows from position 20 of slot 1
	const int C = 40;
	pod::BatchRows pb = pod::BatchRows::causal( C, 20, 1 );
	pod::BatchRows* dprows = up_rows( pb );
	std::vector<half_t> pq = rh( rng, static_cast<size_t>( C ) * HQ * HD, 1.0f ), pg = rh( rng, pq.size(), 2.0f );
	half_t* dpq = up( pq );
	half_t* dpg = up( pg );
	half_t* dpout = zeros<half_t>( pq.size() );
	pod::TensorView pqt = tv( dpq, pod::DType::FP16, C, HQ * HD ), pgt = tv( dpg, pod::DType::FP16, C, HQ * HD ), pot = tv( dpout, pod::DType::FP16, C, HQ * HD );
	Attention::forward_prefill( kv, pqt, pgt, pot, slot_stride, A_LAYER, dprows, C, HQ, HKV, HD, A_SEQ );
	backend::device_sync();
	emit( "attn_prefill_gate", dl( dpout, pq.size() ) );

	// Q / gate unpack: [rows][heads][2 * head_dim] -> q, gate
	std::vector<half_t> qg = rh( rng, static_cast<size_t>( R ) * HQ * 2 * HD, 1.0f );
	half_t* dqg = up( qg );
	half_t* uq = zeros<half_t>( hq.size() );
	half_t* ug = zeros<half_t>( hq.size() );
	pod::TensorView qgt = tv( dqg, pod::DType::FP16, R, HQ * 2 * HD ), uqt = tv( uq, pod::DType::FP16, R, HQ * HD ), ugt = tv( ug, pod::DType::FP16, R, HQ * HD );
	Attention::unpack_q_gate( qgt, uqt, ugt, HQ, HD );
	backend::device_sync();
	emit( "attn_unpack_q", dl( uq, hq.size() ) );
	emit( "attn_unpack_gate", dl( ug, hq.size() ) );
}

// ---- MLA latent: 16 heads, 512 latent ----
constexpr int LH = 16, LORA = 512, LHD = 256;

void test_latent() {
	std::printf( "MLA latent (%d heads, %d latent)\n", LH, LORA );
	GoldenRng rng( 2002 );
	const int max_seq = 256, N = 200, layers = 2, layer = 1;
	const size_t slot_stride = static_cast<size_t>( layers ) * max_seq * LORA;
	half_t* cache = zeros<half_t>( slot_stride );
	std::vector<half_t> hc = rh( rng, static_cast<size_t>( N ) * LORA, 1.7f );
	half_t* dc = up( hc );
	pod::BatchRows sb = pod::BatchRows::causal( N, 0, 0 );
	pod::BatchRows* srows = up_rows( sb );
	pod::SlotRef ssr;
	ssr.rows = srows;
	ssr.row = 0;
	ssr.stride = slot_stride;
	pod::TensorView ct = tv( dc, pod::DType::FP16, N, LORA );
	MLALatent::store( cache, ct, layer, max_seq, N, ssr, nullptr );
	backend::device_sync();
	emit( "latent_cache_layer", dl( cache + static_cast<size_t>( layer ) * max_seq * LORA, static_cast<size_t>( N ) * LORA ) );

	const int R = 3;
	const int positions[3] = { 199, 100, 7 };
	const int slots[3] = { 0, 0, 0 };
	pod::BatchRows br = pod::BatchRows::batched( R, slots, positions );
	pod::BatchRows* drows = up_rows( br );
	pod::SlotRef sr;
	sr.rows = drows;
	sr.row = 0;
	sr.stride = slot_stride;
	std::vector<half_t> hw = rh( rng, static_cast<size_t>( LH ) * 2 * LHD * LORA, 0.05f ), hq = rh( rng, static_cast<size_t>( R ) * LH * LHD, 1.5f );
	half_t* kv_b_p = up( hw );
	half_t* dq = up( hq );
	half_t* dqlat = zeros<half_t>( static_cast<size_t>( R ) * LH * LORA );
	half_t* dolat = zeros<half_t>( static_cast<size_t>( R ) * LH * LORA );
	half_t* dout = zeros<half_t>( static_cast<size_t>( R ) * LH * LHD );
	float* ws = zeros<float>( MLALatent::workspace_floats( max_seq, LH ) );
	pod::TensorView kv_b = tv( kv_b_p, pod::DType::FP16, LH * 2 * LHD, LORA ), qt = tv( dq, pod::DType::FP16, R, LH * LHD );
	pod::TensorView qlat = tv( dqlat, pod::DType::FP16, R, LH * LORA ), olat = tv( dolat, pod::DType::FP16, R, LH * LORA ), outt = tv( dout, pod::DType::FP16, R, LH * LHD );
	MLALatent::absorb_q( qt, kv_b, qlat, LH, R, nullptr );
	MLALatent::attend( cache, qlat, olat, layer, max_seq, R, LH, sr, ws, 0, nullptr );
	MLALatent::expand_out( olat, kv_b, outt, LH, R, nullptr );
	backend::device_sync();
	emit( "latent_absorb_q", dl( dqlat, static_cast<size_t>( R ) * LH * LORA ) );
	emit( "latent_attend", dl( dolat, static_cast<size_t>( R ) * LH * LORA ) );
	emit( "latent_expand_out", dl( dout, static_cast<size_t>( R ) * LH * LHD ) );
}

// ---- DSA: projection, scoring, selection, and latent attention over the selected tokens ----
void test_dsa() {
	std::printf( "DSA (%d heads x %d, pool %d, top %d pools)\n", DSA::HEADS, DSA::DIM, DSA::POOL, DSA::SEL_POOLS );
	GoldenRng rng( 3003 );
	const int max_seq = 3072, NP = max_seq / DSA::POOL, layers = 2, layer = 1;
	const int R = 3;
	// Row 0 is tie-free, row 1 is full of ties, row 2 is below the selection threshold (dense)
	const int positions[3] = { 2998, 3071, 1500 };
	const int slots[3] = { 0, 0, 0 };
	pod::BatchRows br = pod::BatchRows::batched( R, slots, positions );
	pod::BatchRows* drows = up_rows( br );

	// project_small: [rows, k] x [n, k]^T
	{
		const int n = DSA::PROJ, k = 512;
		std::vector<half_t> w = rh( rng, static_cast<size_t>( n ) * k, 0.1f ), x = rh( rng, static_cast<size_t>( R ) * k, 1.0f );
		half_t* dw = up( w );
		half_t* dx = up( x );
		half_t* dp = zeros<half_t>( static_cast<size_t>( R ) * n );
		DSA::project_small( dw, dx, dp, R, n, k, nullptr );
		backend::device_sync();
		emit( "dsa_project_small", dl( dp, static_cast<size_t>( R ) * n ) );
	}

	// score
	{
		const size_t pool_slot_stride = static_cast<size_t>( layers ) * NP * DSA::DIM;
		std::vector<half_t> pool = rh( rng, pool_slot_stride, 1.0f ), q = rh( rng, static_cast<size_t>( R ) * DSA::HEADS * DSA::DIM, 0.5f ), w = rh( rng, static_cast<size_t>( R ) * DSA::HEADS, 1.0f );
		half_t* dpool = up( pool );
		half_t* dq = up( q );
		half_t* dw = up( w );
		half_t* dsc = zeros<half_t>( static_cast<size_t>( R ) * NP );
		DSA::score( dq, dw, dpool, dsc, NP, layer, max_seq, R, drows, pool_slot_stride, nullptr );
		backend::device_sync();
		emit( "dsa_score", dl( dsc, static_cast<size_t>( R ) * NP ) );
	}

	// select on a synthetic score field
	std::vector<half_t> sc( static_cast<size_t>( R ) * NP );
	{
		std::vector<int> perm( NP );
		for ( int p = 0; p < NP; ++p ) perm[p] = p;
		for ( int p = NP - 1; p > 0; --p ) std::swap( perm[p], perm[rng.next() % ( p + 1 )] );
		for ( int p = 0; p < NP; ++p ) {
			sc[p] = backend::from_float_half( ( perm[p] + 1 ) * 0.5f );          // distinct, exact in FP16
			sc[NP + p] = backend::from_float_half( ( ( p * 7919 ) % 61 ) * 0.25f ); // 61 levels: ties at the threshold
			sc[2 * NP + p] = backend::from_float_half( ( perm[p] % 97 ) * 0.1f );
		}
	}
	half_t* dsc = up( sc );
	int* didx = nullptr;
	int* dnk = nullptr;
	{
		std::vector<int> init( static_cast<size_t>( R ) * DSA::KPAD, -7 ), nk( R, -5 );
		didx = up( init );
		dnk = up( nk );
	}
	DSA::select( dsc, NP, didx, dnk, R, drows, nullptr );
	backend::device_sync();
	emit( "dsa_select_idx", dli( didx, static_cast<size_t>( R ) * DSA::KPAD ) );
	emit( "dsa_select_nkeys", dli( dnk, R ) );

	// Latent attention over the selected tokens (rows 0, 1) next to a dense row (2)
	{
		const size_t slot_stride = static_cast<size_t>( layers ) * max_seq * LORA;
		std::vector<half_t> cache = rh( rng, slot_stride, 1.7f ), q = rh( rng, static_cast<size_t>( R ) * LH * LORA, 1.0f );
		half_t* dcache = up( cache );
		half_t* dq = up( q );
		half_t* dout = zeros<half_t>( q.size() );
		float* ws = zeros<float>( MLALatent::workspace_floats( max_seq, LH ) );
		pod::SlotRef sr;
		sr.rows = drows;
		sr.row = 0;
		sr.stride = slot_stride;
		pod::TensorView qt = tv( dq, pod::DType::FP16, R, LH * LORA ), ot = tv( dout, pod::DType::FP16, R, LH * LORA );
		MLALatent::attend( dcache, qt, ot, layer, max_seq, R, LH, sr, ws, 0, nullptr, didx, dnk );
		backend::device_sync();
		emit( "latent_attend_sparse", dl( dout, q.size() ) );
	}
}

// ---- mHC collapse + update ----
void test_mhc() {
	std::printf( "mHC collapse + update\n" );
	GoldenRng rng( 4004 );
	const int H = 4096, R = 3, iters = 20;
	const float eps = 1e-6f;
	std::vector<float> streams = rf( rng, static_cast<size_t>( R ) * 4 * H, 1.0f ), fn = rf( rng, static_cast<size_t>( 24 ) * 4 * H, 0.02f ), base = rf( rng, 24, 0.5f );
	std::vector<float> scale = { 0.5f, 0.7f, 0.9f };
	float* ds = up( streams );
	float* dfn = up( fn );
	float* dbase = up( base );
	float* dscale = up( scale );
	half_t* dcol = zeros<half_t>( static_cast<size_t>( R ) * H );
	float* dpost = zeros<float>( R * 4 );
	float* dcomb = zeros<float>( R * 16 );
	float* dinv = zeros<float>( R );
	float* dmix = zeros<float>( R * 24 );
	float* dpre = zeros<float>( R * 4 );
	pod::TensorView tstreams = tv( ds, pod::DType::FP32, R, 4 * H ), tfn = tv( dfn, pod::DType::FP32, 24, 4 * H ), tbase = tv( dbase, pod::DType::FP32, 1, 24 ), tscale = tv( dscale, pod::DType::FP32, 1, 3 );
	pod::TensorView tcol = tv( dcol, pod::DType::FP16, R, H ), tpost = tv( dpost, pod::DType::FP32, R, 4 ), tcomb = tv( dcomb, pod::DType::FP32, R, 16 );
	pod::TensorView tinv = tv( dinv, pod::DType::FP32, 1, R ), tmix = tv( dmix, pod::DType::FP32, R, 24 ), tpre = tv( dpre, pod::DType::FP32, R, 4 );
	HyperConnection::collapse( tstreams, tfn, tbase, tscale, tcol, tpost, tcomb, tinv, tmix, tpre, H, iters, eps, nullptr, R );
	backend::device_sync();
	emit( "mhc_collapsed", dl( dcol, static_cast<size_t>( R ) * H ) );
	emit( "mhc_post", dlf( dpost, R * 4 ) );
	emit( "mhc_comb", dlf( dcomb, R * 16 ) );
	emit( "mhc_inv_rms", dlf( dinv, R ) );
	emit( "mhc_mix_logits", dlf( dmix, R * 24 ) );
	emit( "mhc_pre", dlf( dpre, R * 4 ) );

	std::vector<half_t> sub = rh( rng, static_cast<size_t>( R ) * H, 1.0f );
	half_t* dsub = up( sub );
	pod::TensorView tsub = tv( dsub, pod::DType::FP16, R, H );
	HyperConnection::update( tstreams, tsub, tpost, tcomb, H, nullptr, R );
	backend::device_sync();
	emit( "mhc_update_streams", dlf( ds, static_cast<size_t>( R ) * 4 * H ) );
}

// ---- GDN single-row forms and the KDA pieces ----
void test_gdn() {
	std::printf( "GDN / KDA\n" );
	GoldenRng rng( 5005 );
	// Single-row causal conv + recurrent step (Qwen shape: 16 K heads, 32 V heads, head_dim 128), slot 2 of 3
	{
		const int hk = 16, hv = 32, hd = 128, channels = ( 2 * hk + hv ) * hd, slots = 3;
		const size_t conv_stride = static_cast<size_t>( channels ) * 3;
		std::vector<float> hist = rf( rng, conv_stride * slots, 0.5f );
		std::vector<half_t> qkv = rh( rng, channels, 1.0f ), w = rh( rng, static_cast<size_t>( channels ) * 4, 0.5f );
		float* dhist = up( hist );
		half_t* dqkv = up( qkv );
		half_t* dw = up( w );
		const int sl[1] = { 2 }, ps[1] = { 0 };
		pod::BatchRows br = pod::BatchRows::batched( 1, sl, ps );
		pod::SlotRef sr;
		sr.rows = up_rows( br );
		sr.row = 0;
		sr.stride = conv_stride;
		pod::TensorView tq = tv( dqkv, pod::DType::FP16, 1, channels ), th = tv( dhist, pod::DType::FP32, slots, static_cast<int>( conv_stride ) ), tw = tv( dw, pod::DType::FP16, channels, 4 );
		GDN::causal_conv1d_silu( tq, th, tw, channels, nullptr, sr, nullptr, nullptr );
		backend::device_sync();
		emit( "gdn_conv_out", dl( dqkv, channels ) );
		emit( "gdn_conv_history", dlf( dhist, conv_stride * slots ) );

		const size_t state_stride = static_cast<size_t>( hv ) * hd * hd;
		std::vector<float> state = rf( rng, state_stride * slots, 0.1f ), gate = rf( rng, hv, 0.4f, -0.6f ), beta = rf( rng, hv, 0.4f, 0.5f );
		std::vector<half_t> rq = rh( rng, static_cast<size_t>( channels ), 0.15f );
		float* dstate = up( state );
		float* dgate = up( gate );
		float* dbeta = up( beta );
		half_t* drq = up( rq );
		half_t* dout = zeros<half_t>( static_cast<size_t>( hv ) * hd );
		pod::SlotRef ssr = sr;
		ssr.stride = state_stride;
		pod::TensorView trq = tv( drq, pod::DType::FP16, 1, channels ), tg = tv( dgate, pod::DType::FP32, 1, hv ), tb = tv( dbeta, pod::DType::FP32, 1, hv ), to = tv( dout, pod::DType::FP16, 1, hv * hd );
		GDN::recurrent_step( dstate, trq, tg, tb, to, hk, hv, hd, nullptr, ssr, nullptr, nullptr );
		backend::device_sync();
		emit( "gdn_step_out", dl( dout, static_cast<size_t>( hv ) * hd ) );
		emit( "gdn_step_state_slot2", dlf( dstate + 2 * state_stride, state_stride ) );
	}

	// KDA gate projection (hidden 4096, 16 heads x 128)
	const int H = 4096, NH = 16, KD = 128, CH = NH * KD;
	{
		const int rows = 2;
		std::vector<half_t> x = rh( rng, static_cast<size_t>( rows ) * H, 1.0f ), bp = rh( rng, static_cast<size_t>( 16 ) * H, 0.05f );
		std::vector<half_t> fa = rh( rng, static_cast<size_t>( 128 ) * H, 0.05f ), ga = rh( rng, static_cast<size_t>( 128 ) * H, 0.05f );
		std::vector<half_t> fb = rh( rng, static_cast<size_t>( CH ) * 128, 0.1f ), gb = rh( rng, static_cast<size_t>( CH ) * 128, 0.1f );
		std::vector<half_t> alog = rh( rng, NH, 0.5f ), dtb = rh( rng, CH, 0.5f );
		half_t* dx = up( x );
		half_t* dbp = up( bp );
		half_t* dfa = up( fa );
		half_t* dga = up( ga );
		half_t* dfb = up( fb );
		half_t* dgb = up( gb );
		half_t* dal = up( alog );
		half_t* ddt = up( dtb );
		float* dbeta = zeros<float>( rows * 16 );
		float* dgd = zeros<float>( static_cast<size_t>( rows ) * CH );
		half_t* dz = zeros<half_t>( static_cast<size_t>( rows ) * CH );
		half_t* dfab = zeros<half_t>( rows * 128 );
		half_t* dgab = zeros<half_t>( rows * 128 );
		pod::TensorView tx = tv( dx, pod::DType::FP16, rows, H ), tbp = tv( dbp, pod::DType::FP16, 16, H ), tfa = tv( dfa, pod::DType::FP16, 128, H ), tga = tv( dga, pod::DType::FP16, 128, H );
		pod::TensorView tfb = tv( dfb, pod::DType::FP16, CH, 128 ), tgb = tv( dgb, pod::DType::FP16, CH, 128 ), tal = tv( dal, pod::DType::FP16, 1, NH ), tdt = tv( ddt, pod::DType::FP16, 1, CH );
		pod::TensorView tbeta = tv( dbeta, pod::DType::FP32, rows, 16 ), tgd = tv( dgd, pod::DType::FP32, rows, CH ), tz = tv( dz, pod::DType::FP16, rows, CH );
		pod::TensorView tfab = tv( dfab, pod::DType::FP16, rows, 128 ), tgab = tv( dgab, pod::DType::FP16, rows, 128 );
		GDN::project_kda_gates( tx, tbp, tfa, tfb, tga, tgb, tal, tdt, tbeta, tgd, tz, tfab, tgab, NH, KD, H, nullptr, rows );
		backend::device_sync();
		emit( "kda_beta", dlf( dbeta, rows * 16 ) );
		emit( "kda_g_decay", dlf( dgd, static_cast<size_t>( rows ) * CH ) );
		emit( "kda_z", dl( dz, static_cast<size_t>( rows ) * CH ) );
		emit( "kda_f_a", dl( dfab, rows * 128 ) );
	}

	// KDA recurrence: rows over two slots, the fused single step and the unfused step
	{
		const int rows = 3, nslots = 2;
		const size_t state_stride = static_cast<size_t>( NH ) * KD * KD;
		std::vector<float> state = rf( rng, state_stride * nslots, 0.05f ), gd = rf( rng, static_cast<size_t>( rows ) * CH, 0.5f, -0.55f ), beta = rf( rng, rows * NH, 0.4f, 0.5f );
		std::vector<half_t> qkv = rh( rng, static_cast<size_t>( rows ) * 3 * CH, 0.1f ), z = rh( rng, static_cast<size_t>( rows ) * CH, 1.0f ), on = rh( rng, KD, 0.1f );
		for ( auto& v : on ) v = backend::from_float_half( 1.0f + backend::to_float( v ) );
		float* dstate = up( state );
		float* dgd = up( gd );
		float* dbeta = up( beta );
		half_t* dqkv = up( qkv );
		half_t* dz = up( z );
		half_t* don = up( on );
		half_t* dout = zeros<half_t>( static_cast<size_t>( rows ) * CH );
		const int sl[3] = { 1, 1, 0 }, ps[3] = { 0, 1, 0 };
		pod::BatchRows br = pod::BatchRows::batched( rows, sl, ps );
		pod::BatchRows* drows = up_rows( br );
		pod::TensorView tqkv = tv( dqkv, pod::DType::FP16, rows, 3 * CH ), tgd = tv( dgd, pod::DType::FP32, rows, CH ), tb = tv( dbeta, pod::DType::FP32, rows, NH );
		pod::TensorView tz = tv( dz, pod::DType::FP16, rows, CH ), ton = tv( don, pod::DType::FP16, 1, KD ), to = tv( dout, pod::DType::FP16, rows, CH );
		GDN::recurrent_step_kda_rows( dstate, tqkv, tgd, tb, tz, ton, to, NH, KD, 1e-5f, rows, drows, state_stride, nullptr, 0, 0, nullptr, nullptr );
		backend::device_sync();
		emit( "kda_rows_out", dl( dout, static_cast<size_t>( rows ) * CH ) );
		emit( "kda_rows_state", dlf( dstate, state_stride * nslots ) );

		// Fused single step on row 0 of the batch, slot 1 of a fresh state copy
		float* dstate2 = up( state );
		half_t* dout2 = zeros<half_t>( CH );
		pod::SlotRef sr;
		sr.rows = drows;
		sr.row = 0;
		sr.stride = state_stride;
		pod::TensorView to2 = tv( dout2, pod::DType::FP16, 1, CH );
		GDN::recurrent_step_kda_fused( dstate2, tqkv, tgd, tb, tz, ton, to2, NH, KD, 1e-5f, nullptr, sr );
		backend::device_sync();
		emit( "kda_fused_out", dl( dout2, CH ) );
		emit( "kda_fused_state_slot1", dlf( dstate2 + state_stride, state_stride ) );

		// Unfused step (no slot, no norm)
		float* dstate3 = up( state );
		half_t* dout3 = zeros<half_t>( CH );
		pod::TensorView to3 = tv( dout3, pod::DType::FP16, 1, CH );
		GDN::recurrent_step_kda( dstate3, tqkv, tgd, tb, to3, NH, KD, nullptr );
		backend::device_sync();
		emit( "kda_step_out", dl( dout3, CH ) );
		emit( "kda_step_state", dlf( dstate3, state_stride ) );
	}
}

}

int main( int argc, char** argv ) {
	GoldenArgs args( argc, argv );
	g_args = &args;
	if ( !args.dump && !args.check ) {
		std::printf( "usage: test_ops_attention --dump <dir> | --check <dir>\n" );
		return 2;
	}
	backend::set_device( 0 );
	std::printf( "test_ops_attention (%s, dir %s)\n", args.dump ? "dump" : "check", args.dir.c_str() );
	test_attention();
	test_latent();
	test_dsa();
	test_mhc();
	test_gdn();
	backend::device_sync();
	for ( void* p : g_allocs ) backend::device_free( p );
	std::printf( "%d outputs, %d failures\n", g_cases, g_fail );
	return g_fail == 0 ? 0 : 1;
}
