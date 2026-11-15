// DeepSeek-V4.1 decode attention (DSV4::attend with its split-KV combine) on one GPU: time per layer over 40 distinct layers' caches (cycled so
// the L2 does not hold them), the window of 128 plus nsel selected compressed entries, as one rank's heads meet a decode step.
//   bench_dsv4_attend [context=1100] [rows=1] [heads=16] [nsel=context/4, at most 512; 0 = a window-only layer]
// BENCH_DSPARK=1 times DSV4::dspark_attend instead: a drafter block of `rows` rows over the window and the block's own keys. Layer 0's output is
// compared with the same call without a workspace (one block a row, no split).
#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/batch.h"
#include "engine/modules/dsv4.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using engine::modules::DSV4;

int main( int argc, char** argv ) {
	const int ctx = argc > 1 ? std::atoi( argv[1] ) : 1100, rows = argc > 2 ? std::atoi( argv[2] ) : 1, nh = argc > 3 ? std::atoi( argv[3] ) : 16;
	const int nsel = argc > 4 ? std::atoi( argv[4] ) : std::min( DSV4::TOPK, ctx / 4 );
	const int LAYERS = 40, entries = std::max( 1, ctx / 4 + rows );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	uint32_t seed = 7u;
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return ( ( seed >> 8 ) / 16777216.0f ) * 2.0f - 1.0f; };
	auto upload_half = [&]( size_t n, float scale ) {
		std::vector<half> h( n );
		for ( auto& v : h ) v = __float2half( rnd() * scale );
		half* d = nullptr;
		CUDA_CHECK( cudaMalloc( &d, n * sizeof( half ) ) );
		CUDA_CHECK( cudaMemcpy( d, h.data(), n * sizeof( half ), cudaMemcpyHostToDevice ) );
		return d;
	};
	const size_t ring_stride = static_cast<size_t>( DSV4::RING ) * DSV4::HD, comp_stride = static_cast<size_t>( entries ) * DSV4::HD;
	std::vector<half*> q( LAYERS ), ring( LAYERS ), comp( LAYERS ), o( LAYERS );
	for ( int l = 0; l < LAYERS; ++l ) {
		q[l] = upload_half( static_cast<size_t>( rows ) * nh * DSV4::HD, 1.0f );
		ring[l] = upload_half( ring_stride, 1.0f );
		comp[l] = upload_half( comp_stride, 1.0f );
		CUDA_CHECK( cudaMalloc( &o[l], static_cast<size_t>( rows ) * nh * DSV4::HD * sizeof( half ) ) );
	}
	// the selected entries in ascending order, the sinks, and a rotary table up to the last position
	std::vector<int> hsel( static_cast<size_t>( rows ) * DSV4::TOPK, 0 ), hn( rows, nsel );
	for ( int r = 0; r < rows; ++r ) for ( int i = 0; i < nsel; ++i ) hsel[static_cast<size_t>( r ) * DSV4::TOPK + i] = i * ( entries - 1 ) / std::max( 1, nsel );
	int *sel = nullptr, *nsel_d = nullptr;
	CUDA_CHECK( cudaMalloc( &sel, hsel.size() * sizeof( int ) ) );
	CUDA_CHECK( cudaMalloc( &nsel_d, hn.size() * sizeof( int ) ) );
	CUDA_CHECK( cudaMemcpy( sel, hsel.data(), hsel.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( nsel_d, hn.data(), hn.size() * sizeof( int ), cudaMemcpyHostToDevice ) );
	std::vector<float> hsink( nh ), hrope( static_cast<size_t>( ctx + rows ) * DSV4::ROPE );
	for ( auto& v : hsink ) v = rnd();
	for ( size_t p = 0; p < static_cast<size_t>( ctx + rows ); ++p )
		for ( int i = 0; i < DSV4::ROPE / 2; ++i ) {
			const float a = static_cast<float>( p ) * std::pow( 10000.0f, -2.0f * i / DSV4::ROPE );
			hrope[p * DSV4::ROPE + 2 * i] = std::cos( a );
			hrope[p * DSV4::ROPE + 2 * i + 1] = std::sin( a );
		}
	float *sink = nullptr, *rope = nullptr, *ws = nullptr;
	CUDA_CHECK( cudaMalloc( &sink, nh * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &rope, hrope.size() * sizeof( float ) ) );
	CUDA_CHECK( cudaMalloc( &ws, DSV4::attend_ws_floats( nh ) * sizeof( float ) ) );
	CUDA_CHECK( cudaMemcpy( sink, hsink.data(), nh * sizeof( float ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( rope, hrope.data(), hrope.size() * sizeof( float ), cudaMemcpyHostToDevice ) );
	pod::BatchRows br = pod::BatchRows::causal( rows, ctx - 1, 0 );
	pod::BatchRows* d_rows = nullptr;
	CUDA_CHECK( cudaMalloc( &d_rows, sizeof( pod::BatchRows ) ) );
	CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( pod::BatchRows ), cudaMemcpyHostToDevice ) );
	const bool dspark = std::getenv( "BENCH_DSPARK" ) != nullptr;
	const int t = ctx - 2, w0 = std::max( 0, t - DSV4::WINDOW + 1 );
	auto run = [&]() {
		for ( int l = 0; l < LAYERS; ++l ) {
			if ( dspark ) DSV4::dspark_attend( q[l], ring[l], comp[l], sink, rope, o[l], nh, 0, rows, d_rows, 0, w0, t, ring_stride, ws, stream );
			else DSV4::attend( q[l], ring[l], nsel > 0 ? comp[l] : nullptr, sel, nsel_d, sink, rope, o[l], nh, 0, rows, d_rows, ring_stride, comp_stride, ws, stream );
		}
	};
	run();
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	cudaGraph_t g;
	cudaGraphExec_t ex;
	CUDA_CHECK( cudaStreamBeginCapture( stream, cudaStreamCaptureModeGlobal ) );
	run();
	CUDA_CHECK( cudaStreamEndCapture( stream, &g ) );
	CUDA_CHECK( cudaGraphInstantiate( &ex, g, nullptr, nullptr, 0 ) );
	CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );
	cudaEvent_t e0, e1;
	CUDA_CHECK( cudaEventCreate( &e0 ) );
	CUDA_CHECK( cudaEventCreate( &e1 ) );
	const int iters = 50;
	CUDA_CHECK( cudaEventRecord( e0, stream ) );
	for ( int i = 0; i < iters; ++i ) CUDA_CHECK( cudaGraphLaunch( ex, stream ) );
	CUDA_CHECK( cudaEventRecord( e1, stream ) );
	CUDA_CHECK( cudaEventSynchronize( e1 ) );
	float ms = 0.0f;
	CUDA_CHECK( cudaEventElapsedTime( &ms, e0, e1 ) );
	std::vector<half> ho( static_cast<size_t>( rows ) * nh * DSV4::HD );
	CUDA_CHECK( cudaMemcpy( ho.data(), o[0], ho.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
	double sum = 0.0;
	for ( auto v : ho ) sum += std::fabs( __half2float( v ) );
	// layer 0 against one block a row (no workspace: the kernel without splits)
	half* ref = nullptr;
	CUDA_CHECK( cudaMalloc( &ref, ho.size() * sizeof( half ) ) );
	if ( dspark ) DSV4::dspark_attend( q[0], ring[0], comp[0], sink, rope, ref, nh, 0, rows, d_rows, 0, w0, t, ring_stride, nullptr, stream );
	else DSV4::attend( q[0], ring[0], nsel > 0 ? comp[0] : nullptr, sel, nsel_d, sink, rope, ref, nh, 0, rows, d_rows, ring_stride, comp_stride, nullptr, stream );
	std::vector<half> hr( ho.size() );
	CUDA_CHECK( cudaMemcpy( hr.data(), ref, hr.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
	double dot = 0.0, na = 0.0, nb = 0.0, maxd = 0.0;
	size_t same = 0;
	for ( size_t i = 0; i < ho.size(); ++i ) {
		const double a = __half2float( ho[i] ), b = __half2float( hr[i] );
		dot += a * b;
		na += a * a;
		nb += b * b;
		maxd = std::max( maxd, std::fabs( a - b ) );
		same += a == b;
	}
	std::printf( "%scontext %d, rows %d, heads %d, %d selected: %.2f us per layer (graph replay over %d layers); checksum %.6f\n", dspark ? "dspark, " : "", ctx, rows, nh, nsel,
		ms * 1000.0f / iters / LAYERS, LAYERS, sum );
	std::printf( "against one block a row: cos %.6f, max |d| %.4g, %.1f%% identical\n", dot / std::sqrt( na * nb ), maxd, 100.0 * same / ho.size() );
	return 0;
}
