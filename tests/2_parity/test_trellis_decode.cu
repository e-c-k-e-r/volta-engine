// The trellis pair decode (decode_3inst_2, codebook.cuh; on ROCm its RDNA form in src/kernels/rocm/trellis/codebook_rdna.cuh) against the
// 32-bit multiply form of the three codebooks, for every 16-bit index in both slots of the pair, with junk in the upper input bits that the
// callers mask off. The result must match bit for bit.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include "codebook.cuh"

namespace {

template <int cb>
__device__ uint32_t reference( uint32_t x0, uint32_t x1 ) {
	x0 &= 0xffffu;
	x1 &= 0xffffu;
	if constexpr ( cb == 0 ) {
		x0 = x0 * 89226354u + 64248484u;
		x1 = x1 * 89226354u + 64248484u;
		EXL3_CB_LOP3( x0 );
		EXL3_CB_LOP3( x1 );
		half2_uint32 a( x0 ), b( x1 );
		half2_uint32 r( __hadd2( __lows2half2( a.as_half2, b.as_half2 ), __highs2half2( a.as_half2, b.as_half2 ) ) );
		return r.as_uint32;
	}
	if constexpr ( cb == 1 ) {
		half2_uint32 r( decode_mcg_product_2( x0 * 0xCBAC1FEDu, x1 * 0xCBAC1FEDu ) );
		return r.as_uint32;
	}
	half2_uint32 r( decode_mul1_product_2( x0 * 0x83DCD12Du, x1 * 0x83DCD12Du ) );
	return r.as_uint32;
}

template <int cb>
__global__ void check_kernel( unsigned* mismatches ) {
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;   // 0 .. 65535
	const uint32_t junk = ( i * 2654435761u ) & 0xffff0000u;
	const uint32_t x0 = i | junk, x1 = ( ( i * 40503u + 7u ) & 0xffffu ) | ( junk ^ 0xa5a50000u );
	const half2_uint32 g( decode_3inst_2<cb>( x0 & 0xffffu, x1 & 0xffffu ) );
	const half2_uint32 g2( decode_3inst_2<cb>( x1 & 0xffffu, x0 & 0xffffu ) );
	if ( g.as_uint32 != reference<cb>( x0, x1 ) ) atomicAdd( mismatches, 1u );
	if ( g2.as_uint32 != reference<cb>( x1, x0 ) ) atomicAdd( mismatches, 1u );
}

template <int cb>
unsigned run( unsigned* d ) {
	unsigned h = 0;
	cudaMemset( d, 0, sizeof( unsigned ) );
	check_kernel<cb><<<256, 256>>>( d );
	cudaMemcpy( &h, d, sizeof( unsigned ), cudaMemcpyDeviceToHost );
	return h;
}

}   // namespace

int main() {
	unsigned* d = nullptr;
	if ( cudaMalloc( &d, sizeof( unsigned ) ) != cudaSuccess ) {
		std::printf( "[-] no GPU\n" );
		return 1;
	}
	const unsigned bad[3] = { run<0>( d ), run<1>( d ), run<2>( d ) };
	const cudaError_t err = cudaDeviceSynchronize();
	cudaFree( d );
	bool ok = err == cudaSuccess;
	for ( int cb = 0; cb < 3; ++cb ) {
		std::printf( "  codebook %d: %u mismatches of 131072 pairs\n", cb, bad[cb] );
		ok &= bad[cb] == 0;
	}
	std::printf( ok ? "[+] trellis decode PASS\n" : "[-] trellis decode FAIL\n" );
	return ok ? 0 : 1;
}
