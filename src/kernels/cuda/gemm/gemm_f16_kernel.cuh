#pragma once
// The device code of the dense FP16 GEMM on Volta's tensor cores (gemm_f16.cu's kernels; also the PTX entry points of gemm_f16_vk.cu that the
// Vulkan backend launches through VK_NV_cuda_kernel_launch): out[m, n] = beta * out + a[m, k] * w[n, k]^T (tn) or a[m, k] * w[k, n] (nn), see
// gemm_f16.cu for the design. Self-contained (cuda_fp16.h only), so the PTX build needs nothing of the engine
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace engine::modules::gemm_f16 {

// B row-major (k rows of 8 n): a lane holds k row lane % 4, columns 4 (lane >= 16) + 0..3
__device__ __forceinline__ void mma884_rr( float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1 ) {
	asm( "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
		: "+f"( d[0] ), "+f"( d[1] ), "+f"( d[2] ), "+f"( d[3] ), "+f"( d[4] ), "+f"( d[5] ), "+f"( d[6] ), "+f"( d[7] ) : "r"( a0 ), "r"( a1 ), "r"( b0 ), "r"( b1 ) );
}

__device__ __forceinline__ void mma884( float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1 ) {
	asm( "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
		: "+f"( d[0] ), "+f"( d[1] ), "+f"( d[2] ), "+f"( d[3] ), "+f"( d[4] ), "+f"( d[5] ), "+f"( d[6] ), "+f"( d[7] ) : "r"( a0 ), "r"( a1 ), "r"( b0 ), "r"( b1 ) );
}

// One operand's k step: rows [r0, r0 + ROWS) of a row-major [rows, k] matrix, values [k0, k0 + BK), as 16-byte chunks (zero past the edges)
template <int ROWS, int BK, int THREADS>
struct Stage {
	static constexpr int kBK = BK;
	// 16-byte chunk c of row r sits at chunk c ^ swizzle( r ): the 8 rows a quarter-warp reads (and the rows it stores) cover all 32 banks
	static __device__ __forceinline__ int swz( int r, int c ) { return c ^ ( ( r / ( 64 / BK ) ) % ( BK / 8 ) ); }
	static constexpr int chunks = ROWS * kBK / 8, per_thread = ( chunks + THREADS - 1 ) / THREADS;
	uint4 v[per_thread];
	__device__ __forceinline__ void load( const half* __restrict__ p, int ld, int rows, int k, int r0, int k0 ) {
		#pragma unroll
		for ( int i = 0; i < per_thread; ++i ) {
			const int idx = threadIdx.x + i * THREADS, r = idx / ( kBK / 8 ), c = 8 * ( idx % ( kBK / 8 ) );
			if ( chunks % THREADS != 0 && idx >= chunks ) break;
			v[i] = r0 + r < rows && k0 + c < k ? __ldg( reinterpret_cast<const uint4*>( p + static_cast<size_t>( r0 + r ) * ld + k0 + c ) ) : make_uint4( 0, 0, 0, 0 );
		}
	}
	__device__ __forceinline__ void store( half* s ) const {
		#pragma unroll
		for ( int i = 0; i < per_thread; ++i ) {
			const int idx = threadIdx.x + i * THREADS, r = idx / ( kBK / 8 ), c = 8 * ( idx % ( kBK / 8 ) );
			if ( chunks % THREADS != 0 && idx >= chunks ) break;
			*reinterpret_cast<uint4*>( s + r * kBK + 8 * swz( r, c / 8 ) ) = v[i];
		}
	}
};

// The weights' k step when they are stored [k, n] (n contiguous): rows [k0, k0 + BK), columns [n0, n0 + BN) as 16-byte chunks (zero past the
// edges, n a multiple of 8); 16-byte chunk c of k row r sits at chunk c ^ ( r % 4 ), so the 4 k rows a quad reads fall in different banks
template <int BK, int BN, int THREADS>
struct StageKN {
	static constexpr int chunks = BK * BN / 8, per_thread = ( chunks + THREADS - 1 ) / THREADS;
	uint4 v[per_thread];
	__device__ __forceinline__ void load( const half* __restrict__ p, int ld, int n, int k, int n0, int k0 ) {
		#pragma unroll
		for ( int i = 0; i < per_thread; ++i ) {
			const int idx = threadIdx.x + i * THREADS, r = idx / ( BN / 8 ), c = 8 * ( idx % ( BN / 8 ) );
			if ( chunks % THREADS != 0 && idx >= chunks ) break;
			v[i] = k0 + r < k && n0 + c < n ? __ldg( reinterpret_cast<const uint4*>( p + static_cast<size_t>( k0 + r ) * ld + n0 + c ) ) : make_uint4( 0, 0, 0, 0 );
		}
	}
	__device__ __forceinline__ void store( half* s ) const {
		#pragma unroll
		for ( int i = 0; i < per_thread; ++i ) {
			const int idx = threadIdx.x + i * THREADS, r = idx / ( BN / 8 ), c = 8 * ( idx % ( BN / 8 ) );
			if ( chunks % THREADS != 0 && idx >= chunks ) break;
			*reinterpret_cast<uint4*>( s + r * BN + 8 * ( ( c / 8 ) ^ ( r & 3 ) ) ) = v[i];
		}
	}
};

template <bool WKN> constexpr int w_stage_halves( int BN, int BK ) { return BN * BK; }

// Row (of A) or column (of W) within a 16-wide fragment that a quad pair's row / column r of its 8 x 8 product maps to: quad pairs 0 and 1
// (2 and 3) split the fragment's rows (columns) as 0-3 + 8-11 and 4-7 + 12-15, so each half-warp reads 8 consecutive rows (no bank conflicts)
__device__ __forceinline__ int frag_line( int half_id, int r ) { return ( r & 3 ) + 4 * half_id + 8 * ( r >> 2 ); }

// Split along k (gridDim.z > 1): each split's partial sums go to part[z][m][n] (FP32), the reduce adds them in split order
// WKN: the weights are stored [k, n] (out = a * w) instead of [n, k] (out = a * w^T)
template <int BM, int BN, int BK, int WM, int WN, bool F32, bool WKN>
__device__ __forceinline__ void gemm_tn_body( const half* __restrict__ a, int lda, const half* __restrict__ w, int ldw, void* __restrict__ out, int ldc, int m, int n,
	int k, float beta, float* __restrict__ part, int k_per_split ) {
	constexpr int THREADS = WM * WN * 32, TM = BM / WM, TN = BN / WN, MB = TM / 16, NB = TN / 16;
	constexpr int kBK = BK;
	extern __shared__ __align__( 16 ) half smem[];
	half( *as )[BM][BK] = reinterpret_cast<half( * )[BM][BK]>( smem );
	half* ws = smem + 2 * BM * BK;
	constexpr int WSTAGE = w_stage_halves<WKN>( BN, BK );
	using SA = Stage<BM, BK, WM * WN * 32>;
	using SW = std::conditional_t<WKN, StageKN<BK, BN, WM * WN * 32>, Stage<BN, BK, WM * WN * 32>>;
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp / WN, wn = warp % WN;
	const int q = ( lane >> 2 ) & 3, qm = q & 1, qn = q >> 1, p = ( lane & 3 ) + 4 * ( lane >> 4 );
	const int arow = wm * TM + frag_line( qm, p ), wrow = wn * TN + frag_line( qn, p );
	const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
	// this split's k range: the operands start at its first value, k is its length
	const int kz0 = blockIdx.z * k_per_split;
	a += kz0;
	w += WKN ? static_cast<size_t>( kz0 ) * ldw : kz0;
	k = min( k_per_split, k - kz0 );
	SA sa;
	SW sw;
	sa.load( a, lda, m, k, m0, 0 );
	sw.load( w, ldw, n, k, n0, 0 );
	sa.store( &as[0][0][0] );
	sw.store( ws );
	__syncthreads();
	float acc[MB][NB][8] = {};
	constexpr int KK = kBK / 4;   // mma k steps per tile
	// a fragment row's swizzle is the same for all of a lane's fragments (they are 16 rows apart)
	const uint32_t afrag = static_cast<uint32_t>( __cvta_generic_to_shared( &as[0][arow][0] ) );
	// [k, n] weights: k row lane % 4 of the step, the quad pair's columns 4 qn + 8 (lane >= 16) + 0..3 of each 16-column fragment (chunk
	// wn TN / 8 + 2 j + (lane >= 16), swizzled by the row)
	const uint32_t wfrag = static_cast<uint32_t>( __cvta_generic_to_shared( WKN ? ws + ( lane & 3 ) * BN + 4 * qn : ws + wrow * BK ) );
	const int wchunk = wn * TN / 8 + ( lane >> 4 );
	const int akey = SA::swz( arow, 0 ), wkey = SA::swz( wrow, 0 );
	uint2 av[2][MB], bv[2][NB];
	auto lds = []( uint2& v, uint32_t addr ) { asm volatile( "ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"( v.x ), "=r"( v.y ) : "r"( addr ) ); };
	auto frags = [&]( int stage, int kk, int buf ) {
		const int ao = 2 * ( 8 * ( ( kk >> 1 ) ^ akey ) + 4 * ( kk & 1 ) ), wo = 2 * ( 8 * ( ( kk >> 1 ) ^ wkey ) + 4 * ( kk & 1 ) );
		#pragma unroll
		for ( int i = 0; i < MB; ++i ) lds( av[buf][i], afrag + 2 * ( stage * BM * BK + 16 * i * BK ) + ao );
		#pragma unroll
		for ( int j = 0; j < NB; ++j ) {
			if constexpr ( WKN ) lds( bv[buf][j], wfrag + 2 * ( stage * WSTAGE + 4 * kk * BN + 8 * ( ( wchunk + 2 * j ) ^ ( lane & 3 ) ) ) );
			else lds( bv[buf][j], wfrag + 2 * ( stage * WSTAGE + 16 * j * BK ) + wo );
		}
	};
	frags( 0, 0, 0 );
	const int steps = ( k + kBK - 1 ) / kBK;
	if ( steps > 1 ) {
		sa.load( a, lda, m, k, m0, kBK );
		sw.load( w, ldw, n, k, n0, kBK );
	}
	for ( int s = 0; s < steps; ++s ) {
		const int cur = s & 1;
		const bool more = s + 1 < steps;
		#pragma unroll
		for ( int kk = 0; kk < KK; ++kk ) {
			// the next step's tiles load during the first k step; the next fragments load while this one multiplies, and before the step's
			// last the next step's tiles go to the other buffer (its last reads were before the previous barrier) for its first fragments
			if ( kk + 1 < KK ) frags( cur, kk + 1, ( kk + 1 ) & 1 );
			else if ( more ) {
				sa.store( &as[cur ^ 1][0][0] );
				sw.store( ws + ( cur ^ 1 ) * WSTAGE );
				__syncthreads();
				frags( cur ^ 1, 0, 0 );
				if ( s + 2 < steps ) {
					sa.load( a, lda, m, k, m0, ( s + 2 ) * kBK );
					sw.load( w, ldw, n, k, n0, ( s + 2 ) * kBK );
				}
			}
			const int b = kk & 1;
			#pragma unroll
			for ( int i = 0; i < MB; ++i ) {
				#pragma unroll
				for ( int j = 0; j < NB; ++j ) {
					if constexpr ( WKN ) mma884_rr( acc[i][j], av[b][i].x, av[b][i].y, bv[b][j].x, bv[b][j].y );
					else mma884( acc[i][j], av[b][i].x, av[b][i].y, bv[b][j].x, bv[b][j].y );
				}
			}
		}
	}
	// Epilogue through shared memory, one 16-row fragment block of the warp's tile at a time: the accumulators go to a per-warp FP32 stage,
	// then each lane writes 8 consecutive columns of a row (16-byte stores where the row and alignment allow)
	__syncthreads();   // the last step's fragments are read: the tiles' memory is free
	constexpr int SLD = TN + 4;
	static_assert( WM * WN * 16 * SLD * 4 <= 2 * ( BM * BK + WSTAGE ) * 2, "epilogue stage fits in the tiles' memory" );
	float* stage = reinterpret_cast<float*>( smem ) + warp * 16 * SLD;
	const int r0 = ( lane & 1 ) + 4 * ( lane >> 4 ), c0 = lane & 2;
	constexpr int LPR = TN / 8, RPP = 32 / LPR;   // lanes per row, rows per pass
	const int er = lane / LPR, ec = 8 * ( lane % LPR );
	const bool split = gridDim.z > 1;
	#pragma unroll
	for ( int i = 0; i < MB; ++i ) {
		#pragma unroll
		for ( int j = 0; j < NB; ++j ) {
			#pragma unroll
			for ( int h = 0; h < 2; ++h ) {
				#pragma unroll
				for ( int rr = 0; rr < 2; ++rr ) {
					const int lr = frag_line( qm, r0 + 2 * rr ), lc = 16 * j + frag_line( qn, c0 + 4 * h );
					*reinterpret_cast<float2*>( stage + lr * SLD + lc ) = make_float2( acc[i][j][4 * h + 2 * rr], acc[i][j][4 * h + 2 * rr + 1] );
				}
			}
		}
		__syncwarp();
		#pragma unroll
		for ( int pr = 0; pr < 16; pr += RPP ) {
			const int lr = pr + er, row = m0 + wm * TM + 16 * i + lr, col = n0 + wn * TN + ec;
			if ( row >= m || col >= n ) continue;
			float v[8];
			*reinterpret_cast<float4*>( v ) = *reinterpret_cast<const float4*>( stage + lr * SLD + ec );
			*reinterpret_cast<float4*>( v + 4 ) = *reinterpret_cast<const float4*>( stage + lr * SLD + ec + 4 );
			const int cnt = min( 8, n - col );
			if ( split ) {
				float* pp = part + ( static_cast<size_t>( blockIdx.z ) * m + row ) * n + col;
				if ( cnt == 8 && ( reinterpret_cast<uintptr_t>( pp ) & 15 ) == 0 ) {
					reinterpret_cast<float4*>( pp )[0] = *reinterpret_cast<float4*>( v );
					reinterpret_cast<float4*>( pp )[1] = *reinterpret_cast<float4*>( v + 4 );
				}
				else for ( int e = 0; e < cnt; ++e ) pp[e] = v[e];
				continue;
			}
			const size_t o = static_cast<size_t>( row ) * ldc + col;
			if constexpr ( F32 ) {
				float* po = static_cast<float*>( out ) + o;
				if ( cnt == 8 && ( reinterpret_cast<uintptr_t>( po ) & 15 ) == 0 ) {
					float4* p4 = reinterpret_cast<float4*>( po );
					if ( beta != 0.0f ) {
						const float4 a0 = p4[0], a1 = p4[1];
						v[0] += beta * a0.x; v[1] += beta * a0.y; v[2] += beta * a0.z; v[3] += beta * a0.w;
						v[4] += beta * a1.x; v[5] += beta * a1.y; v[6] += beta * a1.z; v[7] += beta * a1.w;
					}
					p4[0] = *reinterpret_cast<float4*>( v );
					p4[1] = *reinterpret_cast<float4*>( v + 4 );
				}
				else for ( int e = 0; e < cnt; ++e ) po[e] = beta != 0.0f ? v[e] + beta * po[e] : v[e];
			}
			else {
				half* po = static_cast<half*>( out ) + o;
				if ( cnt == 8 && ( reinterpret_cast<uintptr_t>( po ) & 15 ) == 0 ) {
					uint4 pk;
					half2* h2 = reinterpret_cast<half2*>( &pk );
					if ( beta != 0.0f ) {
						const uint4 old = *reinterpret_cast<const uint4*>( po );
						const half2* o2 = reinterpret_cast<const half2*>( &old );
						#pragma unroll
						for ( int e = 0; e < 4; ++e ) { const float2 f = __half22float2( o2[e] ); v[2 * e] += beta * f.x; v[2 * e + 1] += beta * f.y; }
					}
					#pragma unroll
					for ( int e = 0; e < 4; ++e ) h2[e] = __floats2half2_rn( v[2 * e], v[2 * e + 1] );
					*reinterpret_cast<uint4*>( po ) = pk;
				}
				else for ( int e = 0; e < cnt; ++e ) po[e] = __float2half_rn( beta != 0.0f ? v[e] + beta * __half2float( po[e] ) : v[e] );
			}
		}
		__syncwarp();
	}

}

// out = beta * out + the sum of the splits' partials, 4 columns per thread
template <bool F32>
__device__ __forceinline__ void split_reduce_body( const float* __restrict__ part, void* __restrict__ out, int ldc, int m, int n, int splits, float beta ) {
	const int quads = ( n + 3 ) / 4, i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= m * quads ) return;
	const int row = i / quads, col = 4 * ( i % quads ), cnt = min( 4, n - col );
	const size_t plane = static_cast<size_t>( m ) * n;
	const float* pp = part + static_cast<size_t>( row ) * n + col;
	float v[4] = {};
	if ( cnt == 4 && ( n & 3 ) == 0 ) {
		for ( int z = 0; z < splits; ++z ) {
			const float4 t = *reinterpret_cast<const float4*>( pp + z * plane );
			v[0] += t.x; v[1] += t.y; v[2] += t.z; v[3] += t.w;
		}
	}
	else {
		for ( int z = 0; z < splits; ++z ) {
			for ( int e = 0; e < cnt; ++e ) v[e] += pp[z * plane + e];
		}
	}
	const size_t o = static_cast<size_t>( row ) * ldc + col;
	if constexpr ( F32 ) {
		float* po = static_cast<float*>( out ) + o;
		for ( int e = 0; e < cnt; ++e ) po[e] = beta != 0.0f ? v[e] + beta * po[e] : v[e];
	}
	else {
		half* po = static_cast<half*>( out ) + o;
		if ( cnt == 4 && beta == 0.0f && ( reinterpret_cast<uintptr_t>( po ) & 7 ) == 0 ) {
			half2 h[2] = { __floats2half2_rn( v[0], v[1] ), __floats2half2_rn( v[2], v[3] ) };
			*reinterpret_cast<uint2*>( po ) = *reinterpret_cast<const uint2*>( h );
		}
		else for ( int e = 0; e < cnt; ++e ) po[e] = __float2half_rn( beta != 0.0f ? v[e] + beta * __half2float( po[e] ) : v[e] );
	}
}

} // namespace engine::modules::gemm_f16
