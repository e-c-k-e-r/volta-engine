// Decode-sized sparse-list attention (Attention::forward_sparse, head_dim 256, rows <= 8) against an FP64 softmax over the same keys:
// lists of 1500 / 37 / 256 / 257 keys, a dense row (nkeys < 0), GQA 8 / 2, FP16 KV; outputs are the gated attention of the reduce kernel
// (no gate here). Cosine >= 0.9999 and max abs error against the FP64 result.
#include "golden_io.h"
#include "engine/backend/runtime.h"
#include "engine/backend/numeric.h"
#include "engine/core/batch.h"
#include "engine/core/kv_format.h"
#include "engine/core/tensor.h"
#include "engine/modules/attention.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace engine;
using backend::half_t;

namespace {

template <typename T>
T* up( const std::vector<T>& v ) {
	T* p = static_cast<T*>( backend::device_malloc( std::max<size_t>( v.size() * sizeof( T ), 16 ) ) );
	backend::memcpy_sync( p, v.data(), v.size() * sizeof( T ), backend::Copy::H2D );
	return p;
}

std::vector<half_t> rh( GoldenRng& r, size_t n, float scale ) {
	std::vector<half_t> v( n );
	for ( auto& x : v ) x = backend::from_float_half( r.uniform( scale ) );
	return v;
}

} // namespace

int main() {
	const int HQ = 8, HKV = 2, HD = 256, MAXSEQ = 4096, LAYERS = 2, LAYER = 1, SLOTS = 2, KPAD = 2048 + 16;
	const int R = 5;
	const int slots[R] = { 0, 1, 0, 1, 0 };
	const int pos[R] = { 3000, 2500, 900, 1200, 700 };
	const int nk_host[R] = { 1500, 37, 256, 257, -1 };   // row 4 attends densely to 0 .. pos
	GoldenRng rng( 4242 );
	const size_t slot_stride = static_cast<size_t>( LAYERS ) * MAXSEQ * HKV * HD;
	std::vector<half_t> hk = rh( rng, SLOTS * slot_stride, 1.0f ), hv = rh( rng, SLOTS * slot_stride, 1.0f ), hq = rh( rng, static_cast<size_t>( R ) * HQ * HD, 0.3f );
	std::vector<int> hidx( static_cast<size_t>( R ) * KPAD, -1 );
	for ( int r = 0; r < R; ++r ) {
		if ( nk_host[r] < 0 ) continue;
		std::vector<int> all( pos[r] + 1 );
		for ( int i = 0; i <= pos[r]; ++i ) all[i] = i;
		for ( int i = pos[r]; i > 0; --i ) std::swap( all[i], all[rng.next() % ( i + 1 )] );
		for ( int i = 0; i < nk_host[r]; ++i ) hidx[static_cast<size_t>( r ) * KPAD + i] = all[i];
	}
	half_t *dk = up( hk ), *dv = up( hv ), *dq = up( hq );
	int* didx = up( hidx );
	int* dnk = up( std::vector<int>( nk_host, nk_host + R ) );
	pod::BatchRows br = pod::BatchRows::batched( R, slots, pos );
	pod::BatchRows* drows = static_cast<pod::BatchRows*>( backend::device_malloc( sizeof( br ) ) );
	backend::memcpy_sync( drows, &br, sizeof( br ), backend::Copy::H2D );
	half_t* dout = static_cast<half_t*>( backend::device_malloc( static_cast<size_t>( R ) * HQ * HD * sizeof( half_t ) ) );
	pod::KVRef kv = pod::kv_ref_f16( dk, dv );
	pod::TensorView qt = pod::make_tensor_2d( dq, pod::DType::FP16, R, HQ * HD ), gt = pod::make_tensor_2d( nullptr, pod::DType::FP16, R, HQ * HD ), ot = pod::make_tensor_2d( dout, pod::DType::FP16, R, HQ * HD );
	modules::Attention::forward_sparse( kv, qt, gt, ot, slot_stride, LAYER, drows, R, HQ, HKV, HD, MAXSEQ, didx, dnk, KPAD, nullptr );
	backend::device_sync();
	std::vector<half_t> hout( static_cast<size_t>( R ) * HQ * HD );
	backend::memcpy_sync( hout.data(), dout, hout.size() * sizeof( half_t ), backend::Copy::D2H );

	const double scale = 1.0 / std::sqrt( static_cast<double>( HD ) );
	int failures = 0;
	for ( int r = 0; r < R; ++r ) {
		double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0;
		const int total = nk_host[r] >= 0 ? nk_host[r] : pos[r] + 1;
		for ( int h = 0; h < HQ; ++h ) {
			const int kvh = h / ( HQ / HKV );
			std::vector<double> sc( total );
			double mx = -1e300;
			for ( int i = 0; i < total; ++i ) {
				const int t = nk_host[r] >= 0 ? hidx[static_cast<size_t>( r ) * KPAD + i] : i;
				const size_t off = static_cast<size_t>( slots[r] ) * slot_stride + ( static_cast<size_t>( LAYER ) * MAXSEQ + t ) * HKV * HD + kvh * HD;
				double s = 0.0;
				for ( int d = 0; d < HD; ++d ) s += static_cast<double>( backend::to_float( hq[( static_cast<size_t>( r ) * HQ + h ) * HD + d] ) ) * backend::to_float( hk[off + d] );
				sc[i] = s * scale;
				mx = std::max( mx, sc[i] );
			}
			double l = 0.0;
			std::vector<double> o( HD, 0.0 );
			for ( int i = 0; i < total; ++i ) {
				const int t = nk_host[r] >= 0 ? hidx[static_cast<size_t>( r ) * KPAD + i] : i;
				const size_t off = static_cast<size_t>( slots[r] ) * slot_stride + ( static_cast<size_t>( LAYER ) * MAXSEQ + t ) * HKV * HD + kvh * HD;
				const double p = std::exp( sc[i] - mx );
				l += p;
				for ( int d = 0; d < HD; ++d ) o[d] += p * backend::to_float( hv[off + d] );
			}
			for ( int d = 0; d < HD; ++d ) {
				const double ref = o[d] / l, got = backend::to_float( hout[( static_cast<size_t>( r ) * HQ + h ) * HD + d] );
				dot += ref * got; na += ref * ref; nb += got * got;
				max_abs = std::max( max_abs, std::fabs( ref - got ) );
			}
		}
		const double cosine = dot / std::sqrt( na * nb + 1e-30 );
		const bool ok = cosine >= 0.9999 && max_abs <= 2e-3;
		std::printf( "[%s] row %d (%d keys): cosine %.7f, max abs err %.2e\n", ok ? "+" : "-", r, total, cosine, max_abs );
		failures += ok ? 0 : 1;
	}
	std::printf( failures ? "[-] FAILED (%d)\n" : "[+] PASSED\n", failures );
	return failures ? 1 : 0;
}
