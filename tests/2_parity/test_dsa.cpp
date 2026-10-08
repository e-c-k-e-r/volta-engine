// DSA indexer parity on synthetic data: the GPU chain prep -> pool_update -> score -> select against an FP64 CPU reference of the
// exllamav3 k-pool indexer (LayerNorm'd keys, gate + APE softmax pooling, relu-weighted head sum, top-k pools expanded to tokens plus
// the raw tail), then the sparse latent attention over the selected lists against an FP64 softmax over the same tokens.
#include "engine/config.h"
#include "engine/core/batch.h"
#include "engine/modules/dsa.h"
#include "engine/modules/mla.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

using engine::modules::DSA;
using engine::modules::MLALatent;

static uint32_t g_seed = 99u;
static float rnd() { g_seed = g_seed * 1664525u + 1013904223u; return ( ( g_seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; }

int main() {
	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	constexpr int D = DSA::DIM, H = DSA::HEADS, P = DSA::POOL, LORA = 512;
	const int max_seq = 4096, N = 3500, layers = 2, layer = 1;
	const int NP = max_seq / P;

	std::vector<half> h_proj( static_cast<size_t>( N ) * DSA::PROJ ), h_lnw( D ), h_lnb( D ), h_ape( P * D );
	for ( auto& v : h_proj ) v = __float2half( rnd() * 1.3f );
	for ( auto& v : h_lnw ) v = __float2half( 1.0f + 0.2f * rnd() );
	for ( auto& v : h_lnb ) v = __float2half( 0.1f * rnd() );
	for ( auto& v : h_ape ) v = __float2half( 0.5f * rnd() );

	half *d_proj, *d_lnw, *d_lnb, *d_ape, *d_plane, *d_pool, *d_w, *d_q, *d_scores;
	int *d_idx, *d_nk;
	const size_t plane_stride = static_cast<size_t>( layers ) * DSA::PLANE_CAP * DSA::PLANE, pool_stride = static_cast<size_t>( layers ) * NP * D;
	CUDA_CHECK( cudaMalloc( &d_proj, h_proj.size() * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_lnw, D * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_lnb, D * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_ape, P * D * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_plane, plane_stride * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_pool, pool_stride * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_w, 512 * H * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_q, 512ull * H * D * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_scores, 512ull * NP * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_idx, 512ull * DSA::KPAD * 4 ) );
	CUDA_CHECK( cudaMalloc( &d_nk, 512 * 4 ) );
	CUDA_CHECK( cudaMemset( d_plane, 0, plane_stride * 2 ) );
	CUDA_CHECK( cudaMemset( d_pool, 0, pool_stride * 2 ) );
	CUDA_CHECK( cudaMemcpy( d_proj, h_proj.data(), h_proj.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_lnw, h_lnw.data(), D * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_lnb, h_lnb.data(), D * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_ape, h_ape.data(), P * D * 2, cudaMemcpyHostToDevice ) );
	pod::BatchRows* d_rows;
	CUDA_CHECK( cudaMalloc( &d_rows, sizeof( pod::BatchRows ) ) );

	// Feed the tokens in chunks of 500 (positions straddle pool boundaries between chunks), the last 8 as the query rows
	const int R = 40;
	std::vector<half> h_wout;
	for ( int p0 = 0; p0 < N; ) {
		const int n = ( N - p0 > 500 + R ) ? 500 : ( N - p0 > R ? N - p0 - R : N - p0 );
		pod::BatchRows br = pod::BatchRows::causal( n, p0, 0 );
		CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) );
		DSA::prep( d_proj + static_cast<size_t>( p0 ) * DSA::PROJ, d_lnw, d_lnb, d_plane, d_w, layer, max_seq, n, d_rows, plane_stride, 0 );
		DSA::pool_update( d_plane, d_pool, d_ape, layer, max_seq, n, d_rows, plane_stride, pool_stride, 0 );
		p0 += n;
	}
	CUDA_CHECK( cudaDeviceSynchronize() );
	// Query rows: positions N-R .. N-1, with their own w
	std::vector<half> h_q( static_cast<size_t>( R ) * H * D );
	for ( auto& v : h_q ) v = __float2half( rnd() * 0.9f );
	CUDA_CHECK( cudaMemcpy( d_q, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice ) );
	{
		pod::BatchRows br = pod::BatchRows::causal( R, N - R, 0 );
		CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) );
		DSA::prep( d_proj + static_cast<size_t>( N - R ) * DSA::PROJ, d_lnw, d_lnb, d_plane, d_w, layer, max_seq, R, d_rows, plane_stride, 0 );
		DSA::pool_update( d_plane, d_pool, d_ape, layer, max_seq, R, d_rows, plane_stride, pool_stride, 0 );
		DSA::score( d_q, d_w, d_pool, d_scores, NP, layer, max_seq, R, d_rows, pool_stride, 0 );
		DSA::select( d_scores, NP, d_idx, d_nk, R, d_rows, 0 );
	}
	CUDA_CHECK( cudaDeviceSynchronize() );

	// CPU reference
	std::vector<half> h_plane( plane_stride ), h_pool( pool_stride );
	CUDA_CHECK( cudaMemcpy( h_plane.data(), d_plane, plane_stride * 2, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( h_pool.data(), d_pool, pool_stride * 2, cudaMemcpyDeviceToHost ) );
	h_wout.resize( static_cast<size_t>( R ) * H );
	CUDA_CHECK( cudaMemcpy( h_wout.data(), d_w, h_wout.size() * 2, cudaMemcpyDeviceToHost ) );
	bool ok = true;

	// 1. plane: LayerNorm'd keys and gates
	double plane_err = 0;
	for ( int t = N - 700; t < N; t += 7 ) { // the ring holds the most recent PLANE_CAP positions
		const half* pr = h_proj.data() + static_cast<size_t>( t ) * DSA::PROJ;
		double mean = 0, var = 0;
		for ( int d = 0; d < D; ++d ) mean += __half2float( pr[d] );
		mean /= D;
		for ( int d = 0; d < D; ++d ) var += ( __half2float( pr[d] ) - mean ) * ( __half2float( pr[d] ) - mean );
		var /= D;
		for ( int d = 0; d < D; ++d ) {
			const double ref = ( __half2float( pr[d] ) - mean ) / std::sqrt( var + 1e-6 ) * __half2float( h_lnw[d] ) + __half2float( h_lnb[d] );
			const double got = __half2float( h_plane[( static_cast<size_t>( layer ) * DSA::PLANE_CAP + ( t & ( DSA::PLANE_CAP - 1 ) ) ) * DSA::PLANE + d] );
			plane_err = std::max( plane_err, std::fabs( ref - got ) );
			const double gref = __half2float( pr[D + H + d] );
			const double ggot = __half2float( h_plane[( static_cast<size_t>( layer ) * DSA::PLANE_CAP + ( t & ( DSA::PLANE_CAP - 1 ) ) ) * DSA::PLANE + D + d] );
			plane_err = std::max( plane_err, std::fabs( gref - ggot ) );
		}
	}
	std::printf( "  plane (LayerNorm key + gate) max err %.5f %s\n", plane_err, plane_err < 0.01 ? "ok" : "FAIL" );
	ok = ok && plane_err < 0.01;

	// 2. pools: gate + APE softmax over the members (from the GPU plane, exact members)
	double pool_err = 0;
	const int complete = ( N - R + R ) / P; // pools complete after the last row (N tokens)
	for ( int pl = ( N - 700 ) / P; pl < complete; pl += 3 ) {
		for ( int d = 0; d < D; ++d ) {
			double e[P], sum = 0, acc = 0;
			double mx = -1e30;
			for ( int j = 0; j < P; ++j ) mx = std::max( mx, (double) __half2float( h_plane[( static_cast<size_t>( layer ) * DSA::PLANE_CAP + ( ( pl * P + j ) & ( DSA::PLANE_CAP - 1 ) ) ) * DSA::PLANE + D + d] ) + __half2float( h_ape[j * D + d] ) );
			for ( int j = 0; j < P; ++j ) {
				e[j] = std::exp( __half2float( h_plane[( static_cast<size_t>( layer ) * DSA::PLANE_CAP + ( ( pl * P + j ) & ( DSA::PLANE_CAP - 1 ) ) ) * DSA::PLANE + D + d] ) + __half2float( h_ape[j * D + d] ) - mx );
				sum += e[j];
				acc += e[j] * __half2float( h_plane[( static_cast<size_t>( layer ) * DSA::PLANE_CAP + ( ( pl * P + j ) & ( DSA::PLANE_CAP - 1 ) ) ) * DSA::PLANE + d] );
			}
			const double got = __half2float( h_pool[( static_cast<size_t>( layer ) * NP + pl ) * D + d] );
			pool_err = std::max( pool_err, std::fabs( acc / sum - got ) );
		}
	}
	std::printf( "  pool keys max err %.5f %s\n", pool_err, pool_err < 0.005 ? "ok" : "FAIL" );
	ok = ok && pool_err < 0.005;

	// 3. selection per query row from the GPU pool plane (reference scores in FP64, fp16-rounded)
	std::vector<int> h_idx( static_cast<size_t>( R ) * DSA::KPAD ), h_nk( R );
	CUDA_CHECK( cudaMemcpy( h_idx.data(), d_idx, h_idx.size() * 4, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( h_nk.data(), d_nk, R * 4, cudaMemcpyDeviceToHost ) );
	int total_mismatch = 0;
	for ( int r = 0; r < R; ++r ) {
		const int pos = N - R + r, vis = pos + 1, bound = vis / P, tail = vis % P;
		std::vector<double> sc( bound );
		for ( int pl = 0; pl < bound; ++pl ) {
			double acc = 0;
			for ( int h = 0; h < H; ++h ) {
				double lg = 0;
				for ( int d = 0; d < D; ++d ) lg += (double) __half2float( h_q[( static_cast<size_t>( r ) * H + h ) * D + d] ) * __half2float( h_pool[( static_cast<size_t>( layer ) * NP + pl ) * D + d] );
				acc += std::max( lg, 0.0 ) * __half2float( h_wout[static_cast<size_t>( r ) * H + h] );
			}
			sc[pl] = (double) __half2float( __float2half( (float) ( acc / std::sqrt( (double) D * H ) ) ) );
		}
		std::vector<int> order( bound );
		for ( int i = 0; i < bound; ++i ) order[i] = i;
		std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return sc[a] > sc[b]; } );
		const double thr = sc[order[DSA::SEL_POOLS - 1]];
		std::set<int> ref_tokens;
		for ( int j = 0; j < DSA::SEL_POOLS; ++j ) for ( int i = 0; i < P; ++i ) ref_tokens.insert( order[j] * P + i );
		for ( int i = 0; i < tail; ++i ) ref_tokens.insert( vis - tail + i );
		ENGINE_ASSERT( h_nk[r] == DSA::SEL_POOLS * P + tail, "select: key count" );
		std::set<int> got( h_idx.begin() + static_cast<size_t>( r ) * DSA::KPAD, h_idx.begin() + static_cast<size_t>( r ) * DSA::KPAD + h_nk[r] );
		ENGINE_ASSERT( static_cast<int>( got.size() ) == h_nk[r], "select: duplicate tokens" );
		// pools whose score ties the threshold may legitimately differ; any other difference is an error
		int bad = 0;
		for ( int tkn : got ) if ( !ref_tokens.count( tkn ) && sc[tkn / P < bound ? tkn / P : 0] != thr ) ++bad;
		for ( int tkn : ref_tokens ) if ( !got.count( tkn ) && ( tkn / P >= bound || sc[tkn / P] != thr ) ) ++bad;
		total_mismatch += bad;
		for ( int tkn : got ) if ( tkn > pos ) { std::printf( "  row %d selects future token %d\n", r, tkn ); ++total_mismatch; }
	}
	std::printf( "  selection: %d non-tie mismatches over %d rows %s\n", total_mismatch, R, total_mismatch == 0 ? "ok" : "FAIL" );
	ok = ok && total_mismatch == 0;

	// 4. sparse attention over the selected lists vs FP64 softmax over the same tokens
	half *d_cache, *d_qlat, *d_olat; float* d_ws;
	const size_t slot_stride = static_cast<size_t>( layers ) * max_seq * LORA;
	CUDA_CHECK( cudaMalloc( &d_cache, slot_stride * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_qlat, static_cast<size_t>( R ) * 16 * LORA * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_olat, static_cast<size_t>( R ) * 16 * LORA * 2 ) );
	CUDA_CHECK( cudaMalloc( &d_ws, MLALatent::workspace_floats( max_seq, 16 ) * 4 ) );
	std::vector<half> h_cache( slot_stride ), h_qlat( static_cast<size_t>( R ) * 16 * LORA );
	for ( auto& v : h_cache ) v = __float2half( rnd() * 1.5f );
	for ( auto& v : h_qlat ) v = __float2half( rnd() * 0.3f );
	CUDA_CHECK( cudaMemcpy( d_cache, h_cache.data(), slot_stride * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( d_qlat, h_qlat.data(), h_qlat.size() * 2, cudaMemcpyHostToDevice ) );
	pod::SlotRef sr; sr.rows = d_rows; sr.row = 0; sr.stride = slot_stride;
	pod::TensorView qt = pod::make_tensor_2d( d_qlat, pod::DType::FP16, R, 16 * LORA );
	pod::TensorView ot = pod::make_tensor_2d( d_olat, pod::DType::FP16, R, 16 * LORA );
	for ( int nrows : { 8, R } ) {
		pod::TensorView qt2 = pod::make_tensor_2d( d_qlat, pod::DType::FP16, nrows, 16 * LORA );
		pod::TensorView ot2 = pod::make_tensor_2d( d_olat, pod::DType::FP16, nrows, 16 * LORA );
		MLALatent::attend( d_cache, qt2, ot2, layer, max_seq, nrows, 16, sr, d_ws, 0, 0, d_idx, d_nk );
		CUDA_CHECK( cudaDeviceSynchronize() );
		std::vector<half> h_o( static_cast<size_t>( nrows ) * 16 * LORA );
		CUDA_CHECK( cudaMemcpy( h_o.data(), d_olat, h_o.size() * 2, cudaMemcpyDeviceToHost ) );
		double dot = 0, na = 0, nb = 0, maxabs = 0;
		for ( int r = 0; r < nrows; ++r ) {
			for ( int h = 0; h < 16; ++h ) {
				std::vector<double> s( h_nk[r] );
				double mx = -1e300;
				for ( int j = 0; j < h_nk[r]; ++j ) {
					const int tkn = h_idx[static_cast<size_t>( r ) * DSA::KPAD + j];
					double a = 0;
					for ( int c = 0; c < LORA; ++c ) a += (double) __half2float( h_qlat[( static_cast<size_t>( r ) * 16 + h ) * LORA + c] ) * __half2float( h_cache[( static_cast<size_t>( layer ) * max_seq + tkn ) * LORA + c] );
					s[j] = a / 16.0;
					mx = std::max( mx, s[j] );
				}
				double l = 0;
				for ( auto& v : s ) { v = std::exp( v - mx ); l += v; }
				for ( int c = 0; c < LORA; ++c ) {
					double o = 0;
					for ( int j = 0; j < h_nk[r]; ++j ) o += s[j] * __half2float( h_cache[( static_cast<size_t>( layer ) * max_seq + h_idx[static_cast<size_t>( r ) * DSA::KPAD + j] ) * LORA + c] );
					o /= l;
					const double got = __half2float( h_o[( static_cast<size_t>( r ) * 16 + h ) * LORA + c] );
					dot += o * got; na += o * o; nb += got * got;
					maxabs = std::max( maxabs, std::fabs( o - got ) );
				}
			}
		}
		const double cosine = dot / std::sqrt( na * nb + 1e-30 );
		std::printf( "  sparse attention, %d rows (%s kernel, %d keys/row): cos=%.6f max_abs=%.5f %s\n", nrows, nrows > 16 ? "tensor-core" : "SIMT", h_nk[0], cosine, maxabs, ( cosine >= 0.9999 && maxabs < 0.01 ) ? "ok" : "FAIL" );
		ok = ok && cosine >= 0.9999 && maxabs < 0.01;
	}
	std::printf( "%s DSA parity\n", ok ? "[+] PASS" : "[-] FAIL" );
	return ok ? 0 : 1;
}
