#include "engine/core/arena.h"
#include "engine/modules/rope.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>

// Both pairings against an FP64 CPU reference: contiguous halves (Llama / Qwen) and interleaved pairs (GLM4V / MRoPE).
static bool run( bool interleaved, cudaStream_t stream, pod::MemoryArena& arena ) {
	const int nq = 8, nk = 2, hd = 128, rot = 64, rows = 5;
	const float base = 1000000.0f;
	int sq[2] = { rows, nq * hd }, sk[2] = { rows, nk * hd };
	pod::TensorView dq = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sq, 2 );
	pod::TensorView dk = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, sk, 2 );
	std::vector<half> hq( rows * nq * hd ), hk( rows * nk * hd );
	for ( size_t i = 0; i < hq.size(); ++i ) hq[i] = __float2half_rn( std::sin( 0.013f * i ) );
	for ( size_t i = 0; i < hk.size(); ++i ) hk[i] = __float2half_rn( std::cos( 0.017f * i ) );
	CUDA_CHECK( cudaMemcpy( dq.data, hq.data(), hq.size() * sizeof( half ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( dk.data, hk.data(), hk.size() * sizeof( half ), cudaMemcpyHostToDevice ) );

	pod::BatchRows br = pod::BatchRows::causal( rows, 1000, 0 );
	pod::BatchRows* d_rows = nullptr;
	CUDA_CHECK( cudaMalloc( &d_rows, sizeof( br ) ) );
	CUDA_CHECK( cudaMemcpy( d_rows, &br, sizeof( br ), cudaMemcpyHostToDevice ) );
	engine::modules::RoPE::forward( dq, dk, nq, nk, hd, rot, d_rows, rows, base, stream, interleaved );
	CUDA_CHECK( cudaStreamSynchronize( stream ) );

	std::vector<half> oq( hq.size() ), ok( hk.size() );
	CUDA_CHECK( cudaMemcpy( oq.data(), dq.data, oq.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( ok.data(), dk.data, ok.size() * sizeof( half ), cudaMemcpyDeviceToHost ) );
	cudaFree( d_rows );

	auto check = [&]( const std::vector<half>& in, const std::vector<half>& out, int heads ) {
		double dot = 0, na = 0, nb = 0, worst = 0;
		for ( int r = 0; r < rows; ++r ) for ( int h = 0; h < heads; ++h ) for ( int c = 0; c < hd; ++c ) {
			const size_t o = ( (size_t) r * heads + h ) * hd;
			double ref = __half2float( in[o + c] );
			if ( c < rot ) {
				const int i = interleaved ? c >> 1 : ( c < rot / 2 ? c : c - rot / 2 );
				const double th = ( 1000.0 + r ) * std::pow( (double) base, -2.0 * i / rot );
				const int p0 = interleaved ? 2 * i : i, p1 = interleaved ? 2 * i + 1 : i + rot / 2;
				const double a = __half2float( in[o + p0] ), b = __half2float( in[o + p1] );
				ref = ( c == p0 ) ? a * std::cos( th ) - b * std::sin( th ) : a * std::sin( th ) + b * std::cos( th );
			}
			const double got = __half2float( out[o + c] );
			dot += ref * got; na += ref * ref; nb += got * got;
			worst = std::fmax( worst, std::fabs( ref - got ) );
		}
		const double cosv = dot / std::sqrt( na * nb );
		std::printf( "    cos %.7f max abs err %.5f\n", cosv, worst );
		return cosv >= 0.9999 && worst < 0.01;
	};
	return check( hq, oq, nq ) && check( hk, ok, nk );
}

int main() {
	CUDA_CHECK( cudaSetDevice( 0 ) );
	cudaStream_t stream;
	CUDA_CHECK( cudaStreamCreate( &stream ) );
	pod::MemoryArena arena;
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 16ULL * 1024 * 1024 ), "arena" );
	bool pass = true;
	for ( bool il : { false, true } ) {
		std::printf( "[*] RoPE %s\n", il ? "interleaved" : "contiguous halves" );
		pass = run( il, stream, arena ) && pass;
	}
	std::printf( pass ? "[+] RoPE parity PASS\n" : "[-] RoPE parity FAIL\n" );
	return pass ? 0 : 1;
}
