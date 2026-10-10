// Op-level cross-backend golden test for the Qwen4Exp (Flash-Next) modules: gated residual (fp32 streams: init, mix, deferred apply, the
// tensor-core composition, stream norm, broadcast, sigmoid-gated add), the n-gram embedding layer (PLE), the QSA indexer (prep, pool update,
// score), DSA select over a synthetic score field, the sparse-list attention, and the softmax top-k router.
// ONE source builds in the CUDA configuration (--dump runs the real CUDA kernels and writes golden files) and in the CPU / Vulkan ones
// (--check recomputes and compares); inputs come from GoldenRng fixed seeds.
//   test_ops_flash --dump <dir> | --check <dir> [--only gr,ple,qsa,sparse,router]
#include "golden_io.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/core/tensor.h"
#include "engine/modules/attention.h"
#include "engine/modules/dsa.h"
#include "engine/modules/gated_residual.h"
#include "engine/modules/moe.h"
#include "engine/modules/ple.h"
#include "engine/modules/qsa.h"
#include <algorithm>
#include <vector>

using namespace engine;
using backend::half_t;
using modules::GatedResidual;
using modules::PLE;
using modules::QSA;
using modules::DSA;
using modules::Attention;
using modules::MoERouter;

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

std::vector<float> dli( const int* d, size_t n ) {
	backend::device_sync();
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

// ---- gated residual: hidden 512, 4 streams, rank 64 ----
void test_gated_residual() {
	std::printf( "gated residual (hidden 512, 4 streams, rank 64)\n" );
	GoldenRng rng( 4001 );
	const int R = 5, H = 512, HC = 4, RANK = 64, FLAT = HC * H;
	const float eps = 1e-6f;
	std::vector<half_t> hin = rh( rng, static_cast<size_t>( R ) * H, 1.0f );
	half_t* din = up( hin );
	float* x = zeros<float>( static_cast<size_t>( R ) * FLAT );
	GatedResidual::init_streams( din, x, R, H, HC, nullptr );
	emit( "gr_init_streams", dlf( x, static_cast<size_t>( R ) * FLAT ) );

	half_t* norm_w = up( rh( rng, static_cast<size_t>( HC ) * H, 0.3f ) );
	half_t* down = up( rh( rng, static_cast<size_t>( RANK ) * FLAT, 0.05f ) );
	half_t* upw = up( rh( rng, static_cast<size_t>( HC ) * H * RANK, 0.1f ) );
	half_t* inject = up( rh( rng, static_cast<size_t>( HC ) * FLAT, 0.05f ) );
	std::vector<float> hx = rf( rng, static_cast<size_t>( R ) * FLAT, 2.0f );
	float* xs = up( hx );
	half_t* mixed = zeros<half_t>( static_cast<size_t>( R ) * H );
	float* post = zeros<float>( static_cast<size_t>( R ) * HC );
	float* scratch = zeros<float>( GatedResidual::scratch_floats( R, H, HC, RANK ) );
	GatedResidual::mix( xs, norm_w, down, upw, inject, mixed, post, scratch, R, H, HC, RANK, eps, nullptr );
	emit( "gr_mix_mixed", dl( mixed, static_cast<size_t>( R ) * H ) );
	emit( "gr_mix_post", dlf( post, static_cast<size_t>( R ) * HC ) );

	// mix with a deferred apply folded in and no post gate
	half_t* prev_y = up( rh( rng, static_cast<size_t>( R ) * H, 1.0f ) );
	float* xs2 = up( hx );
	float* post2 = up( std::vector<float>( static_cast<size_t>( R ) * HC, 0.75f ) );
	half_t* mixed2 = zeros<half_t>( static_cast<size_t>( R ) * H );
	GatedResidual::mix( xs2, norm_w, down, upw, nullptr, mixed2, nullptr, scratch, R, H, HC, RANK, eps, nullptr, post2, prev_y );
	emit( "gr_mix_deferred_mixed", dl( mixed2, static_cast<size_t>( R ) * H ) );
	emit( "gr_mix_deferred_streams", dlf( xs2, static_cast<size_t>( R ) * FLAT ) );

	// the tensor-core composition of the same mix
	{
		float* xs3 = up( hx );
		float* normed = zeros<float>( static_cast<size_t>( R ) * FLAT );
		half_t* n_hi = zeros<half_t>( static_cast<size_t>( R ) * FLAT );
		half_t* n_lo = zeros<half_t>( static_cast<size_t>( R ) * FLAT );
		GatedResidual::norm( xs3, norm_w, normed, n_hi, n_lo, R, H, HC, eps, nullptr, nullptr, nullptr );
		emit( "gr_norm_normed", dlf( normed, static_cast<size_t>( R ) * FLAT ) );
		// z = normed @ [down; inject]^T
		std::vector<half_t> cat = rh( rng, static_cast<size_t>( RANK + HC ) * FLAT, 0.05f );
		half_t* dcat = up( cat );
		const int ld = RANK + HC;
		float* z = zeros<float>( static_cast<size_t>( R ) * ld );
		GatedResidual::gemm_f32( n_hi, dcat, z, R, ld, FLAT, false, nullptr );
		GatedResidual::gemm_f32( n_lo, dcat, z, R, ld, FLAT, true, nullptr );
		emit( "gr_gemm_f32", dlf( z, static_cast<size_t>( R ) * ld ) );
		half_t* t_hi = zeros<half_t>( static_cast<size_t>( R ) * RANK );
		half_t* t_lo = zeros<half_t>( static_cast<size_t>( R ) * RANK );
		float* post3 = zeros<float>( static_cast<size_t>( R ) * HC );
		GatedResidual::gate_act( z, ld, t_hi, t_lo, post3, R, RANK, HC, true, nullptr );
		emit( "gr_gate_act_t", dl( t_hi, static_cast<size_t>( R ) * RANK ) );
		emit( "gr_gate_act_post", dlf( post3, static_cast<size_t>( R ) * HC ) );
		float* u = up( rf( rng, static_cast<size_t>( R ) * FLAT, 1.5f ) );
		half_t* mixed3 = zeros<half_t>( static_cast<size_t>( R ) * H );
		GatedResidual::up_epilogue( u, normed, mixed3, R, H, HC, nullptr );
		emit( "gr_up_epilogue", dl( mixed3, static_cast<size_t>( R ) * H ) );
	}

	// apply: x[:, s, :] += post[:, s] * y
	{
		float* xa = up( hx );
		half_t* y = up( rh( rng, static_cast<size_t>( R ) * H, 1.0f ) );
		GatedResidual::apply( xa, post2, y, R, H, HC, nullptr );
		emit( "gr_apply", dlf( xa, static_cast<size_t>( R ) * FLAT ) );
	}
	// per-stream grouped norm, broadcast add, sigmoid-gated add
	{
		half_t* out = zeros<half_t>( static_cast<size_t>( R ) * FLAT );
		GatedResidual::stream_norm( xs, norm_w, out, R, H, HC, eps, nullptr );
		emit( "gr_stream_norm", dl( out, static_cast<size_t>( R ) * FLAT ) );
		half_t* hh = up( rh( rng, static_cast<size_t>( R ) * FLAT, 1.0f ) );
		half_t* ee = up( rh( rng, static_cast<size_t>( R ) * H, 1.0f ) );
		float* st = zeros<float>( static_cast<size_t>( R ) * FLAT );
		GatedResidual::add_broadcast( hh, ee, st, R, H, HC, nullptr );
		emit( "gr_add_broadcast", dlf( st, static_cast<size_t>( R ) * FLAT ) );
		half_t* y = up( rh( rng, static_cast<size_t>( R ) * H, 1.0f ) );
		half_t* sh = up( rh( rng, static_cast<size_t>( R ) * H, 1.0f ) );
		half_t* gx = up( rh( rng, static_cast<size_t>( R ) * H, 1.0f ) );
		half_t* gw = up( rh( rng, H, 0.1f ) );
		GatedResidual::add_sigmoid_gated( y, sh, gx, gw, R, H, nullptr );
		emit( "gr_add_sigmoid_gated", dl( y, static_cast<size_t>( R ) * H ) );
	}
}

// ---- PLE: rows of two slots with earlier chunk rows and ring history ----
void test_ple() {
	std::printf( "PLE (hidden 512, 4 streams, conv dilation 3)\n" );
	GoldenRng rng( 5002 );
	const int R = 6, H = 512, HC = 4, C = HC * H, DIL = 3, SLOTS = 2;
	const float eps = 1e-6f, gate_scale = 0.125f;
	half_t* key = up( rh( rng, static_cast<size_t>( R ) * C, 1.0f ) );
	half_t* value = up( rh( rng, static_cast<size_t>( R ) * H, 1.0f ) );
	float* streams = up( rf( rng, static_cast<size_t>( R ) * C, 1.5f ) );
	half_t* nk = up( rh( rng, static_cast<size_t>( HC ) * H, 0.3f ) );
	half_t* nq = up( rh( rng, static_cast<size_t>( HC ) * H, 0.3f ) );
	half_t* nc = up( rh( rng, static_cast<size_t>( HC ) * H, 0.3f ) );
	float* gated = zeros<float>( static_cast<size_t>( R ) * C );
	half_t* normed = zeros<half_t>( static_cast<size_t>( R ) * C );
	PLE::gate_and_norm( key, value, streams, nk, nq, nc, gated, normed, R, H, HC, eps, gate_scale, nullptr );
	emit( "ple_gated", dlf( gated, static_cast<size_t>( R ) * C ) );
	emit( "ple_normed", dl( normed, static_cast<size_t>( R ) * C ) );

	// slot 0 rows at positions 20..23 (taps 3 apart reach the ring), slot 1 rows at 1 and 2 (taps below 0 are zero)
	const int slots[R] = { 0, 0, 0, 0, 1, 1 };
	const int pos[R] = { 20, 21, 22, 23, 1, 2 };
	pod::BatchRows* drows = up_rows( pod::BatchRows::batched( R, slots, pos ) );
	half_t* ring = up( rh( rng, static_cast<size_t>( SLOTS ) * PLE::RING * C, 1.0f ) );
	half_t* conv_w = up( rh( rng, static_cast<size_t>( C ) * 4, 0.5f ) );
	PLE::conv_apply( normed, conv_w, ring, drows, gated, streams, R, C, DIL, nullptr );
	emit( "ple_conv_streams", dlf( streams, static_cast<size_t>( R ) * C ) );
	PLE::ring_update( normed, ring, drows, R, C, nullptr );
	emit( "ple_ring_slot0", dl( ring + 20 * static_cast<size_t>( C ), 4 * static_cast<size_t>( C ) ) );
}

// ---- QSA indexer ----
constexpr int QMAXP = 640;   // pools per layer: room for positions beyond the 512-pool selection budget

void test_qsa() {
	std::printf( "QSA (%d heads x %d, pool %d)\n", QSA::HEADS, QSA::DIM, QSA::POOL );
	GoldenRng rng( 6003 );
	const int R = 8, ROT = 64, LAYERS = 2, LAYER = 1, SLOTS = 2;
	const float theta = 1.0e7f, eps = 1e-6f;
	// slot 0: rows 7..10 and the pool boundary at 11; slot 1: rows 2..5 (pool 1 completes at 7 from the ring)
	const int slots[R] = { 0, 0, 0, 0, 1, 1, 1, 1 };
	const int pos[R] = { 8, 9, 10, 11, 4, 5, 6, 7 };
	pod::BatchRows* drows = up_rows( pod::BatchRows::batched( R, slots, pos ) );
	half_t* qk = up( rh( rng, static_cast<size_t>( R ) * QSA::PROJ, 1.0f ) );
	half_t* qw = up( rh( rng, QSA::DIM, 0.3f ) );
	half_t* kw = up( rh( rng, QSA::DIM, 0.3f ) );
	half_t* q_out = zeros<half_t>( static_cast<size_t>( R ) * QSA::HEADS * QSA::DIM );
	const size_t ring_slot = static_cast<size_t>( LAYERS ) * QSA::RING * QSA::DIM;
	const size_t pool_slot = static_cast<size_t>( LAYERS ) * QMAXP * QSA::DIM;
	// earlier raw keys of both slots (positions below the rows) so a pool that spans the chunk edge reads them
	half_t* ring = up( rh( rng, SLOTS * ring_slot, 1.0f ) );
	half_t* pool = zeros<half_t>( SLOTS * pool_slot );
	QSA::prep( qk, qw, q_out, ring, LAYER, R, drows, ring_slot, ROT, theta, eps, nullptr );
	emit( "qsa_q", dl( q_out, static_cast<size_t>( R ) * QSA::HEADS * QSA::DIM ) );
	emit( "qsa_ring_row", dl( ring + static_cast<size_t>( LAYER ) * QSA::RING * QSA::DIM + 8 * QSA::DIM, 4 * static_cast<size_t>( QSA::DIM ) ) );
	QSA::pool_update( ring, pool, kw, LAYER, R, drows, ring_slot, pool_slot, QMAXP, ROT, theta, eps, nullptr );
	emit( "qsa_pool_slot0", dl( pool + static_cast<size_t>( LAYER ) * QMAXP * QSA::DIM, 4 * static_cast<size_t>( QSA::DIM ) ) );
	emit( "qsa_pool_slot1", dl( pool + pool_slot + static_cast<size_t>( LAYER ) * QMAXP * QSA::DIM, 4 * static_cast<size_t>( QSA::DIM ) ) );

	// score: rows past the selection budget (512 pools) against random pooled keys
	const int SR = 3;
	const int s_slots[SR] = { 0, 1, 0 };
	const int s_pos[SR] = { 2500, 2300, 1000 };   // the last stays dense (no scores)
	pod::BatchRows* srows = up_rows( pod::BatchRows::batched( SR, s_slots, s_pos ) );
	half_t* pool2 = up( rh( rng, SLOTS * pool_slot, 1.0f ) );
	half_t* q2 = up( rh( rng, static_cast<size_t>( SR ) * QSA::HEADS * QSA::DIM, 0.5f ) );
	half_t* scores = zeros<half_t>( static_cast<size_t>( SR ) * QMAXP );
	QSA::score( q2, pool2, scores, QMAXP, LAYER, QMAXP, SR, srows, pool_slot, nullptr );
	const int bound0 = ( s_pos[0] + 1 ) / QSA::POOL, bound1 = ( s_pos[1] + 1 ) / QSA::POOL;
	const std::vector<float> s0 = dl( scores, QMAXP ), s1 = dl( scores + QMAXP, QMAXP );
	emit( "qsa_score_row0", std::vector<float>( s0.begin(), s0.begin() + bound0 ) );
	emit( "qsa_score_row1", std::vector<float>( s1.begin(), s1.begin() + bound1 ) );
}

// ---- DSA select over a synthetic score field, then attention over the selected lists ----
void test_sparse() {
	const int HQ = 8, HKV = 2, HD = 256, MAXSEQ = 2560, LAYERS = 2, LAYER = 1, SLOTS = 2;
	std::printf( "sparse attention (GQA %d/%d x %d)\n", HQ, HKV, HD );
	GoldenRng rng( 7004 );
	const int R = 3, NP = MAXSEQ / DSA::POOL;
	const int slots[R] = { 0, 1, 0 };
	const int pos[R] = { 2500, 2300, 700 };   // rows 0, 1 select, row 2 is dense
	pod::BatchRows* drows = up_rows( pod::BatchRows::batched( R, slots, pos ) );
	std::vector<half_t> sc( static_cast<size_t>( R ) * NP );
	for ( int r = 0; r < R; ++r ) {
		std::vector<int> perm( NP );
		for ( int p = 0; p < NP; ++p ) perm[p] = p;
		for ( int p = NP - 1; p > 0; --p ) std::swap( perm[p], perm[rng.next() % ( p + 1 )] );
		for ( int p = 0; p < NP; ++p ) sc[static_cast<size_t>( r ) * NP + p] = backend::from_float_half( ( r == 1 ? ( perm[p] % 61 ) * 0.25f : ( perm[p] + 1 ) * 0.5f ) );   // row 1: ties at the threshold
	}
	half_t* dsc = up( sc );
	int* idx = up( std::vector<int>( static_cast<size_t>( R ) * DSA::KPAD, -7 ) );
	int* nkeys = up( std::vector<int>( R, -5 ) );
	DSA::select( dsc, NP, idx, nkeys, R, drows, nullptr );
	emit( "sparse_select_idx", dli( idx, static_cast<size_t>( R ) * DSA::KPAD ) );
	emit( "sparse_select_nkeys", dli( nkeys, R ) );

	const size_t slot_stride = static_cast<size_t>( LAYERS ) * MAXSEQ * HKV * HD;
	half_t* dk = up( rh( rng, SLOTS * slot_stride, 1.0f ) );
	half_t* dv = up( rh( rng, SLOTS * slot_stride, 1.0f ) );
	pod::KVRef kv = pod::kv_ref_f16( dk, dv );
	half_t* q = up( rh( rng, static_cast<size_t>( R ) * HQ * HD, 0.3f ) );
	half_t* gate = up( rh( rng, static_cast<size_t>( R ) * HQ * HD, 1.0f ) );
	half_t* out = zeros<half_t>( static_cast<size_t>( R ) * HQ * HD );
	pod::TensorView qt = tv( q, pod::DType::FP16, R, HQ * HD ), gt = tv( gate, pod::DType::FP16, R, HQ * HD ), ot = tv( out, pod::DType::FP16, R, HQ * HD );
	Attention::forward_sparse( kv, qt, gt, ot, slot_stride, LAYER, drows, R, HQ, HKV, HD, MAXSEQ, idx, nkeys, DSA::KPAD, nullptr );
	emit( "sparse_attention", dl( out, static_cast<size_t>( R ) * HQ * HD ) );
}

// ---- softmax top-k router: 512 experts, 10 routes ----
void test_router() {
	std::printf( "softmax router (512 experts, top 10)\n" );
	GoldenRng rng( 8005 );
	const int R = 4, H = 512, E = 512, K = 10;
	half_t* x = up( rh( rng, static_cast<size_t>( R ) * H, 1.0f ) );
	half_t* w = up( rh( rng, static_cast<size_t>( H ) * E, 0.2f ) );   // [hidden, experts]
	int* idx = zeros<int>( static_cast<size_t>( R ) * K );
	half_t* wt = zeros<half_t>( static_cast<size_t>( R ) * K );
	pod::TensorView xt = tv( x, pod::DType::FP16, R, H ), wtv = tv( w, pod::DType::FP16, H, E );
	MoERouter::route_softmax_topk( xt, wtv, idx, wt, H, E, K, nullptr, R );
	emit( "router_idx", dli( idx, static_cast<size_t>( R ) * K ) );
	emit( "router_weights", dl( wt, static_cast<size_t>( R ) * K ) );
}

} // namespace

int main( int argc, char** argv ) {
	GoldenArgs args( argc, argv );
	g_args = &args;
	if ( !args.dump && !args.check ) {
		std::printf( "usage: test_ops_flash --dump <dir> | --check <dir> [--only gr,ple,qsa,sparse,router]\n" );
		return 2;
	}
	backend::set_device( 0 );
	std::printf( "test_ops_flash (%s, dir %s)\n", args.dump ? "dump" : "check", args.dir.c_str() );
	std::string only;
	for ( int i = 1; i + 1 < argc; ++i ) {
		if ( !std::strcmp( argv[i], "--only" ) ) only = std::string( "," ) + argv[i + 1] + ",";
	}
	auto want = [&]( const char* name ) { return only.empty() || only.find( std::string( "," ) + name + "," ) != std::string::npos; };
	if ( want( "gr" ) ) test_gated_residual();
	if ( want( "ple" ) ) test_ple();
	if ( want( "qsa" ) ) test_qsa();
	if ( want( "sparse" ) ) test_sparse();
	if ( want( "router" ) ) test_router();
	backend::device_sync();
	for ( void* p : g_allocs ) backend::device_free( p );
	std::printf( "%d outputs, %d failures\n", g_cases, g_fail );
	return g_fail == 0 ? 0 : 1;
}
