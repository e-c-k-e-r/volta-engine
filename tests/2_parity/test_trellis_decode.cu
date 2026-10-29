// The trellis pair decode (decode_3inst_2, codebook.cuh; on ROCm its RDNA form in src/kernels/rocm/trellis/codebook_rdna.cuh) against the
// 32-bit multiply form of the three codebooks, for every 16-bit index in both slots of the pair, with junk in the upper input bits that the
// callers mask off; and the tile-plan form of dq_dispatch (src/kernels/rocm/trellis/exl3_tile_rdna.cuh: the lane's words loaded through a
// plan, then decoded) against dq_dispatch, every lane of random tiles at every width. Both must match bit for bit.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include "exl3_dq.cuh"
#include "exl3_tile_rdna.cuh"

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

// One random tile per block (words from a hash of the block index) in global memory, where the plan's loads expect it; one lane per thread
template <int bits, int cb>
__global__ void tile_kernel( unsigned* mismatches, uint32_t* tiles ) {
	constexpr int words = bits > 8 ? 4 * ( 2 * ( bits - 8 ) + 1 ) : 8 * bits;
	uint32_t* tile = tiles + blockIdx.x * 64;
	for ( int i = threadIdx.x; i < words; i += 32 ) {
		uint32_t h = ( blockIdx.x * 64u + i ) * 2654435761u;
		h ^= h >> 15;
		h *= 2246822519u;
		tile[i] = h ^ ( h >> 13 );
	}
	__threadfence_block();
	__syncthreads();
	FragB g0, g1, r0, r1;
	const Exl3TilePlan<bits> plan = exl3_tile_plan<bits>( threadIdx.x );
	uint32_t raw[Exl3TilePlan<bits>::R];
	exl3_tile_load<bits>( reinterpret_cast<const char*>( tile ), plan, raw );
	exl3_tile_decode<bits, cb>( raw, plan, g0, g1 );
	dq_dispatch<bits, cb>( tile, threadIdx.x << 3, r0, r1 );
	bool same = true;
	for ( int i = 0; i < 2; ++i ) {
		same &= half2_uint32( g0[i] ).as_uint32 == half2_uint32( r0[i] ).as_uint32;
		same &= half2_uint32( g1[i] ).as_uint32 == half2_uint32( r1[i] ).as_uint32;
	}
	if ( !same ) atomicAdd( mismatches, 1u );
}

template <int bits, int cb>
unsigned run_tiles( unsigned* d ) {
	static uint32_t* tiles = nullptr;
	if ( tiles == nullptr && cudaMalloc( &tiles, 4096 * 64 * sizeof( uint32_t ) ) != cudaSuccess ) return 1;
	unsigned h = 0;
	cudaMemset( d, 0, sizeof( unsigned ) );
	tile_kernel<bits, cb><<<4096, 32>>>( d, tiles );
	cudaMemcpy( &h, d, sizeof( unsigned ), cudaMemcpyDeviceToHost );
	return h;
}

// Widths 1 to 8 and the half-integer rates 1.5 / 2.5 / 3.5 (width codes 9 to 11, mul1 only)
template <int cb>
unsigned run_widths( unsigned* d ) {
	unsigned n = run_tiles<1, cb>( d ) + run_tiles<2, cb>( d ) + run_tiles<3, cb>( d ) + run_tiles<4, cb>( d ) + run_tiles<5, cb>( d ) +
		run_tiles<6, cb>( d ) + run_tiles<7, cb>( d ) + run_tiles<8, cb>( d );
	if constexpr ( cb == 2 ) n += run_tiles<9, cb>( d ) + run_tiles<10, cb>( d ) + run_tiles<11, cb>( d );
	return n;
}

}   // namespace

int main() {
	unsigned* d = nullptr;
	if ( cudaMalloc( &d, sizeof( unsigned ) ) != cudaSuccess ) {
		std::printf( "[-] no GPU\n" );
		return 1;
	}
	const unsigned bad[3] = { run<0>( d ), run<1>( d ), run<2>( d ) };
	const unsigned bad_tiles[3] = { run_widths<0>( d ), run_widths<1>( d ), run_widths<2>( d ) };
	const cudaError_t err = cudaDeviceSynchronize();
	cudaFree( d );
	bool ok = err == cudaSuccess;
	for ( int cb = 0; cb < 3; ++cb ) {
		std::printf( "  codebook %d: %u mismatches of 131072 pairs, %u in the tile-plan decode (%d widths x 131072 lanes)\n", cb, bad[cb], bad_tiles[cb],
			cb == 2 ? 11 : 8 );
		ok &= bad[cb] == 0 && bad_tiles[cb] == 0;
	}
	std::printf( ok ? "[+] trellis decode PASS\n" : "[-] trellis decode FAIL\n" );
	return ok ? 0 : 1;
}
