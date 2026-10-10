// Gated residual, decode-sized calls: the streaming kernels (rows <= 8) against the row-tile kernels (mix_reference): mixed, post, and the time of
// 96 calls over distinct weight sets (cold L2, the way a decode step meets 48 layers x 2 sublayers), graph replay.
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/modules/gated_residual.h"
#include "engine/backend/runtime.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using engine::modules::GatedResidual;
using half_t = engine::backend::half_t;

namespace {

uint32_t g_seed = 99u;
float rnd() { g_seed = g_seed * 1664525u + 1013904223u; return ( ( g_seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; }

void* dalloc( size_t bytes ) { void* p = nullptr; CUDA_CHECK( cudaMalloc( &p, bytes ) ); return p; }

half_t* upload_half( const std::vector<float>& v ) {
	std::vector<half> h( v.size() );
	for ( size_t i = 0; i < v.size(); ++i ) h[i] = __float2half( v[i] );
	half_t* d = static_cast<half_t*>( dalloc( h.size() * sizeof( half ) ) );
	CUDA_CHECK( cudaMemcpy( d, h.data(), h.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
	return d;
}

} // namespace

int main() {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t s;
	CUDA_CHECK( cudaStreamCreate( &s ) );
	const int H = 2560, hc = 4, flat = hc * H, rank = 320, R = 8, SETS = 96;
	const float eps = 1e-6f;
	std::vector<half_t*> w_norm( SETS ), down( SETS ), up( SETS ), inject( SETS );
	for ( int i = 0; i < SETS; ++i ) {
		std::vector<float> a( static_cast<size_t>( hc ) * H ), d( static_cast<size_t>( rank ) * flat ), u( static_cast<size_t>( flat ) * rank ), j( static_cast<size_t>( hc ) * flat );
		for ( auto& v : a ) v = rnd() * 0.2f;
		for ( auto& v : d ) v = rnd() * 0.02f;
		for ( auto& v : u ) v = rnd() * 0.1f;
		for ( auto& v : j ) v = rnd() * 0.02f;
		w_norm[i] = upload_half( a ); down[i] = upload_half( d ); up[i] = upload_half( u ); inject[i] = upload_half( j );
	}
	float* x = static_cast<float*>( dalloc( sizeof( float ) * R * flat ) );
	float* x2 = static_cast<float*>( dalloc( sizeof( float ) * R * flat ) );
	float* post = static_cast<float*>( dalloc( sizeof( float ) * R * hc ) );
	float* post2 = static_cast<float*>( dalloc( sizeof( float ) * R * hc ) );
	float* scratch = static_cast<float*>( dalloc( sizeof( float ) * GatedResidual::scratch_floats( R, H, hc, rank ) ) );
	float* scratch2 = static_cast<float*>( dalloc( sizeof( float ) * GatedResidual::scratch_floats( R, H, hc, rank ) ) );
	half_t* mixed = static_cast<half_t*>( dalloc( 2ull * R * H ) );
	half_t* mixed2 = static_cast<half_t*>( dalloc( 2ull * R * H ) );
	half_t* prev_y = nullptr;
	{
		std::vector<float> py( static_cast<size_t>( R ) * H );
		for ( auto& v : py ) v = rnd();
		prev_y = upload_half( py );
	}
	float* prev_post = static_cast<float*>( dalloc( sizeof( float ) * R * hc ) );
	std::vector<float> hx( static_cast<size_t>( R ) * flat ), hp( R * hc );
	for ( auto& v : hx ) v = rnd() * 2.0f;
	for ( auto& v : hp ) v = 0.5f + 0.5f * rnd();
	CUDA_CHECK( cudaMemcpy( prev_post, hp.data(), hp.size() * sizeof( float ), cudaMemcpyHostToDevice ) );

	int failures = 0;
	for ( int rows = 1; rows <= 8; ++rows ) {
		for ( int variant = 0; variant < 3; ++variant ) {
			const int use_prev = variant & 1;
			const bool use_inject = variant != 2;   // variant 2: no inject rows (the MTP mixer)
			CUDA_CHECK( cudaMemcpy( x, hx.data(), hx.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			CUDA_CHECK( cudaMemcpy( x2, hx.data(), hx.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
			GatedResidual::mix( x, w_norm[0], down[0], up[0], use_inject ? inject[0] : nullptr, mixed, post, scratch, rows, H, hc, rank, eps, s, use_prev ? prev_post : nullptr, use_prev ? prev_y : nullptr );
			GatedResidual::mix_reference( x2, w_norm[0], down[0], up[0], use_inject ? inject[0] : nullptr, mixed2, post2, scratch2, rows, H, hc, rank, eps, s, use_prev ? prev_post : nullptr, use_prev ? prev_y : nullptr );
			CUDA_CHECK( cudaStreamSynchronize( s ) );
			std::vector<half> a( static_cast<size_t>( rows ) * H ), b( a.size() );
			std::vector<float> pa( rows * hc ), pb( rows * hc ), xa( static_cast<size_t>( rows ) * flat ), xb( xa.size() );
			CUDA_CHECK( cudaMemcpy( a.data(), mixed, a.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( b.data(), mixed2, b.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( pa.data(), post, pa.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( pb.data(), post2, pb.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( xa.data(), x, xa.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			CUDA_CHECK( cudaMemcpy( xb.data(), x2, xb.size() * sizeof( float ), cudaMemcpyDeviceToHost ) );
			double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0, ref_max = 0.0, post_err = 0.0;
			for ( size_t i = 0; i < a.size(); ++i ) {
				const double p = __half2float( a[i] ), q = __half2float( b[i] );
				dot += p * q; na += p * p; nb += q * q;
				max_abs = std::fmax( max_abs, std::fabs( p - q ) );
				ref_max = std::fmax( ref_max, std::fabs( q ) );
			}
			for ( size_t i = 0; use_inject && i < pa.size(); ++i ) post_err = std::fmax( post_err, std::fabs( pa[i] - pb[i] ) );
			bool x_same = true;
			for ( size_t i = 0; i < xa.size(); ++i ) x_same = x_same && xa[i] == xb[i];
			const double cosine = dot / std::sqrt( na * nb + 1e-30 );
			const bool ok = cosine >= 0.99999 && max_abs <= 0.002 * ref_max + 1e-3 && post_err <= 1e-4 && x_same;
			std::printf( "[%s] rows %d prev %d inject %d: cosine %.7f, mixed max abs %.2e (ref max %.2f), post max abs %.2e, x updated identically: %s\n", ok ? "+" : "-", rows, use_prev, use_inject ? 1 : 0, cosine, max_abs, ref_max,
				post_err, x_same ? "yes" : "NO" );
			failures += ok ? 0 : 1;
		}
	}

	// Timing: 96 calls over distinct weights, graph replay (the norm pass is part of mix())
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	for ( int pass = 0; pass < 2; ++pass ) {
		for ( int rows : { 1, 4, 8 } ) {
			auto run = [&]() {
				for ( int i = 0; i < SETS; ++i ) {
					if ( pass == 0 ) GatedResidual::mix_reference( x, w_norm[i], down[i], up[i], inject[i], mixed, post, scratch, rows, H, hc, rank, eps, s, prev_post, prev_y );
					else GatedResidual::mix( x, w_norm[i], down[i], up[i], inject[i], mixed, post, scratch, rows, H, hc, rank, eps, s, prev_post, prev_y );
				}
			};
			run();
			CUDA_CHECK( cudaStreamSynchronize( s ) );
			cudaGraph_t graph;
			cudaGraphExec_t exec;
			CUDA_CHECK( cudaStreamBeginCapture( s, cudaStreamCaptureModeGlobal ) );
			run();
			CUDA_CHECK( cudaStreamEndCapture( s, &graph ) );
			CUDA_CHECK( cudaGraphInstantiate( &exec, graph, nullptr, nullptr, 0 ) );
			CUDA_CHECK( cudaGraphLaunch( exec, s ) );
			CUDA_CHECK( cudaStreamSynchronize( s ) );
			CUDA_CHECK( cudaEventRecord( e0, s ) );
			const int iters = 20;
			for ( int i = 0; i < iters; ++i ) CUDA_CHECK( cudaGraphLaunch( exec, s ) );
			CUDA_CHECK( cudaEventRecord( e1, s ) );
			CUDA_CHECK( cudaEventSynchronize( e1 ) );
			float ms = 0.0f;
			CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
			std::printf( "  %s rows %d: %.1f us per mix (norm + gate + up-mix), %.2f ms per %d sublayers\n", pass == 0 ? "row-tile " : "streaming", rows, ms * 1000.0f / iters / SETS, ms / iters, SETS );
		}
	}
	std::printf( failures ? "[-] FAILED (%d)\n" : "[+] PASSED\n", failures );
	return failures ? 1 : 0;
}
