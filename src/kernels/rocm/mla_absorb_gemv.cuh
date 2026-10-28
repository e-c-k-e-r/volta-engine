// ROCm absorb_q for decode-sized blocks, included by src/backend/cuda/modules/mla_latent.cu after its HD / LORA constants
#pragma once

// Decode-sized absorb_q (rows <= 8): a GEMV per head that streams W_UK row by row (a 16 x 16 WMMA tile of it is 32 byte segments, a 5x slower
// read). A block owns (head, 128 columns): 16 column groups of 8 (one 16-byte load) x 16 interleaved d subsets, partial sums added through shared memory.
__global__ __launch_bounds__( 256 )
void absorb_q_small_kernel( const half* __restrict__ q, const half* __restrict__ kv_b, half* __restrict__ q_lat, int heads, int rows ) {
	__shared__ float sq[8][HD];
	__shared__ float sred[16][16][9];
	const int h = blockIdx.x, c0 = blockIdx.y * 128;
	const int t = threadIdx.x, cg = t & 15, ds = t >> 4;
	for ( int i = t; i < 8 * HD; i += 256 ) {
		const int r = i / HD, d = i % HD;
		sq[r][d] = r < rows ? __half2float( q[( static_cast<size_t>( r ) * heads + h ) * HD + d] ) : 0.0f;
	}
	__syncthreads();
	float acc[8][8];
	#pragma unroll
	for ( int r = 0; r < 8; ++r ) {
		#pragma unroll
		for ( int j = 0; j < 8; ++j ) acc[r][j] = 0.0f;
	}
	const half* w = kv_b + static_cast<size_t>( h ) * 2 * HD * LORA + c0 + cg * 8;
	#pragma unroll 4
	for ( int d = ds; d < HD; d += 16 ) {
		const uint4 raw = *reinterpret_cast<const uint4*>( w + static_cast<size_t>( d ) * LORA );
		const half2* h2 = reinterpret_cast<const half2*>( &raw );
		float f[8];
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const float2 v = __half22float2( h2[i] );
			f[2 * i] = v.x;
			f[2 * i + 1] = v.y;
		}
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) {
			const float a = sq[r][d];
			#pragma unroll
			for ( int j = 0; j < 8; ++j ) acc[r][j] = fmaf( a, f[j], acc[r][j] );
		}
	}
	for ( int r = 0; r < rows; ++r ) {
		#pragma unroll
		for ( int j = 0; j < 8; ++j ) sred[ds][cg][j] = acc[r][j];
		__syncthreads();
		if ( t < 128 ) {
			const int g = t >> 3, j = t & 7;
			float sum = 0.0f;
			#pragma unroll
			for ( int k = 0; k < 16; ++k ) sum += sred[k][g][j];
			q_lat[( ( static_cast<size_t>( h >> 4 ) * rows + r ) * 16 + ( h & 15 ) ) * LORA + c0 + g * 8 + j] = __float2half_rn( sum );
		}
		__syncthreads();
	}
}
