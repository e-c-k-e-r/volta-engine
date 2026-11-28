// The vision towers' bidirectional attention on the Volta mma.m8n8k4: included by vision.cu (a __global__ kernel) and by
// src/kernels/cuda/attention/vision_attn_vk.cu (VISION_ATTN_KERNEL defined __device__ first, the body of the PTX entries the Vulkan backend
// launches through VK_NV_cuda_kernel_launch; the tiles' static shared memory fits a launch's 48 KiB). The including file provides mma884.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#ifndef VISION_ATTN_KERNEL
#define VISION_ATTN_KERNEL __global__ void __launch_bounds__( 128, 2 )
#endif

// Register-resident flash attention on the Volta mma.m8n8k4 (FP32 accumulate) for head_dim a multiple of 8 up to 80. A warp owns 16
// queries whose Q fragments stay in registers; K tiles of 64 keys sit row-major and V transposed in shared memory. Quad pair qp of
// the S = Q K^T mma owns the 8 x 8 tile (rows 8 * (qp >> 1), keys 8 * (qp & 1)). A lane holds 4 columns of 2 rows of it, so the
// softmax statistics reduce over lane bits 1 and 2, and P V needs each lane to hold the 8 key columns of its own row: one
// half-pair exchange with the lane t ^ 2 provides them. The two quad pairs sharing a row block add up different keys of the same
// output tile, merged at the end.
constexpr int kMmaWarps = 4;
constexpr int kMmaKeys = 64;
constexpr int kMmaVs = 80;   // halves per transposed V row: 40 words, 8 mod 32, so a half-warp's 8-byte reads hit distinct banks

template <int HD>
VISION_ATTN_KERNEL attention_mma_kernel( const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v,
	half* __restrict__ out, int m, int heads, float scale ) {
	constexpr int KS = 2 * ( ( HD / 2 + 5 ) / 8 * 8 + 2 );   // halves per K row: words 2 mod 8, so a half-warp's 8-byte reads hit distinct banks
	constexpr int NG = HD / 8;
	__shared__ __align__( 16 ) half sK[kMmaKeys * KS];
	__shared__ __align__( 16 ) half sVt[HD * kMmaVs];
	const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	const int t = lane & 3, hi = lane >> 4, qp = ( lane >> 2 ) & 3, e = t + 4 * hi;
	const int head = blockIdx.y, width = heads * HD;
	const int q_base = blockIdx.x * kMmaWarps * 16 + warp * 16;
	const int bstar = t >> 1;   // which of the lane's two accumulator rows is its A row for P V

	uint2 qa[HD / 4];
	{
		const int qr = q_base + 8 * ( qp >> 1 ) + e;
		#pragma unroll
		for ( int ks = 0; ks < HD / 4; ++ks ) {
			qa[ks] = qr < m ? *reinterpret_cast<const uint2*>( q + static_cast<size_t>( qr ) * width + head * HD + 4 * ks ) : make_uint2( 0, 0 );
		}
	}
	float o[NG][8];
	#pragma unroll
	for ( int g = 0; g < NG; ++g ) {
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) o[g][r] = 0.0f;
	}
	float run_max[2] = { -1e30f, -1e30f }, lsum[2] = { 0.0f, 0.0f };
	const float sl = scale * 1.4426950408889634f;

	for ( int t0 = 0; t0 < m; t0 += kMmaKeys ) {
		__syncthreads();
		for ( int idx = tid; idx < kMmaKeys * ( HD / 8 ); idx += kMmaWarps * 32 ) {
			const int key = idx / ( HD / 8 ), c = idx % ( HD / 8 );
			uint4 raw = make_uint4( 0, 0, 0, 0 );
			if ( t0 + key < m ) raw = *reinterpret_cast<const uint4*>( k + static_cast<size_t>( t0 + key ) * width + head * HD + c * 8 );
			uint2* dst = reinterpret_cast<uint2*>( sK + key * KS + c * 8 );
			dst[0] = make_uint2( raw.x, raw.y );
			dst[1] = make_uint2( raw.z, raw.w );
		}
		for ( int idx = tid; idx < ( kMmaKeys / 2 ) * ( HD / 8 ); idx += kMmaWarps * 32 ) {
			const int kp = idx / ( HD / 8 ), c = idx % ( HD / 8 );
			uint4 ra = make_uint4( 0, 0, 0, 0 ), rb = make_uint4( 0, 0, 0, 0 );
			if ( t0 + 2 * kp < m ) ra = *reinterpret_cast<const uint4*>( v + static_cast<size_t>( t0 + 2 * kp ) * width + head * HD + c * 8 );
			if ( t0 + 2 * kp + 1 < m ) rb = *reinterpret_cast<const uint4*>( v + static_cast<size_t>( t0 + 2 * kp + 1 ) * width + head * HD + c * 8 );
			const half* va = reinterpret_cast<const half*>( &ra );
			const half* vb = reinterpret_cast<const half*>( &rb );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				*reinterpret_cast<half2*>( sVt + ( c * 8 + i ) * kMmaVs + 2 * kp ) = __halves2half2( va[i], vb[i] );
			}
		}
		__syncthreads();

		float c[4][8];
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) c[j][r] = 0.0f;
		}
		#pragma unroll
		for ( int ks = 0; ks < HD / 4; ++ks ) {
			#pragma unroll
			for ( int j = 0; j < 4; ++j ) {
				const uint2 b = *reinterpret_cast<const uint2*>( sK + ( 16 * j + 8 * ( qp & 1 ) + e ) * KS + 4 * ks );
				mma884( c[j], qa[ks].x, qa[ks].y, b.x, b.y );
			}
		}
		// Scale into the log2 domain and mask keys past the end
		float mx[2] = { -1e30f, -1e30f };
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) {
				const int key = t0 + 16 * j + 8 * ( qp & 1 ) + ( r & 1 ) + 2 * ( t >> 1 ) + 4 * ( r >> 2 );
				c[j][r] = key < m ? c[j][r] * sl : -1e30f;
				mx[( r >> 1 ) & 1] = fmaxf( mx[( r >> 1 ) & 1], c[j][r] );
			}
		}
		float corr[2];
		#pragma unroll
		for ( int b = 0; b < 2; ++b ) {
			mx[b] = fmaxf( mx[b], __shfl_xor_sync( 0xffffffff, mx[b], 2 ) );
			mx[b] = fmaxf( mx[b], __shfl_xor_sync( 0xffffffff, mx[b], 4 ) );
			const float nm = fmaxf( run_max[b], mx[b] );
			corr[b] = exp2f( run_max[b] - nm );
			run_max[b] = nm;
			lsum[b] *= corr[b];
		}
		#pragma unroll
		for ( int g = 0; g < NG; ++g ) {
			#pragma unroll
			for ( int r = 0; r < 8; ++r ) o[g][r] *= corr[( r >> 1 ) & 1];
		}
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			half2 X[2], Y[2];
			#pragma unroll
			for ( int b = 0; b < 2; ++b ) {
				const float p0 = exp2f( c[j][2 * b] - run_max[b] ), p1 = exp2f( c[j][2 * b + 1] - run_max[b] );
				const float p2 = exp2f( c[j][4 + 2 * b] - run_max[b] ), p3 = exp2f( c[j][5 + 2 * b] - run_max[b] );
				lsum[b] += p0 + p1 + p2 + p3;
				X[b] = __floats2half2_rn( p0, p1 );
				Y[b] = __floats2half2_rn( p2, p3 );
			}
			// Keep the row this lane feeds as an A operand, trade the other one with the lane t ^ 2
			const half2 own_x = bstar ? X[1] : X[0], own_y = bstar ? Y[1] : Y[0];
			const half2 send_x = bstar ? X[0] : X[1], send_y = bstar ? Y[0] : Y[1];
			const uint32_t rx = __shfl_xor_sync( 0xffffffff, *reinterpret_cast<const uint32_t*>( &send_x ), 2 );
			const uint32_t ry = __shfl_xor_sync( 0xffffffff, *reinterpret_cast<const uint32_t*>( &send_y ), 2 );
			const uint32_t ox = *reinterpret_cast<const uint32_t*>( &own_x ), oy = *reinterpret_cast<const uint32_t*>( &own_y );
			const uint32_t a0[2] = { bstar ? rx : ox, bstar ? ry : oy };
			const uint32_t a1[2] = { bstar ? ox : rx, bstar ? oy : ry };
			#pragma unroll
			for ( int sidx = 0; sidx < 2; ++sidx ) {
				#pragma unroll
				for ( int g = 0; g < NG; ++g ) {
					const uint2 b = *reinterpret_cast<const uint2*>( sVt + ( 8 * g + e ) * kMmaVs + 16 * j + 8 * ( qp & 1 ) + 4 * sidx );
					mma884( o[g], a0[sidx], a1[sidx], b.x, b.y );
				}
			}
		}
	}

	float inv[2];
	#pragma unroll
	for ( int b = 0; b < 2; ++b ) {
		lsum[b] += __shfl_xor_sync( 0xffffffff, lsum[b], 2 );
		lsum[b] += __shfl_xor_sync( 0xffffffff, lsum[b], 4 );
		inv[b] = 1.0f / lsum[b];
	}
	#pragma unroll
	for ( int g = 0; g < NG; ++g ) {
		#pragma unroll
		for ( int r = 0; r < 8; ++r ) o[g][r] += __shfl_xor_sync( 0xffffffff, o[g][r], 4 );
	}
	if ( ( qp & 1 ) == 0 ) {
		#pragma unroll
		for ( int r = 0; r < 8; r += 2 ) {
			const int b = ( r >> 1 ) & 1;
			const int row = q_base + 8 * ( qp >> 1 ) + ( t & 1 ) + 2 * b + 4 * hi;
			if ( row < m ) {
				#pragma unroll
				for ( int g = 0; g < NG; ++g ) {
					const int dim = 8 * g + 2 * ( t >> 1 ) + 4 * ( r >> 2 );
					*reinterpret_cast<half2*>( out + static_cast<size_t>( row ) * width + head * HD + dim ) = __floats2half2_rn( o[g][r] * inv[b], o[g][r + 1] * inv[b] );
				}
			}
		}
	}
}
