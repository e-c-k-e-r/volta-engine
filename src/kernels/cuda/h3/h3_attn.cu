// H3 attention for V100 on WMMA (HMMA.884 underneath): a warp owns 16 query rows, keeps its Q tile as A fragments and its output as FP32 accumulators in registers, and
// walks the keys in tiles of 64: S = Q K^T in registers, the online softmax on the accumulators (using the Volta accumulator layout), P through shared memory as the A
// operand of O += P V. No mask logic outside the last key tile, no GQA, no paging: only what the H3 transformer, its VAE decoder and its text tower need.
#include <bitset>
#include <mutex>
#include "engine/video/h3_attn.h"
#include "engine/config.h"
#include <mma.h>
#include <cstdint>

namespace engine::video {
namespace {

using namespace nvcuda;

constexpr int kD = 128, kBQ = 128, kBK = 64, kWarps = 8, kThreads = kWarps * 32;
constexpr int kKV = kD + 16;  // shared row stride of K / V tiles (halves): the +8 spreads the fragment loads over the banks
constexpr int kPS = kBK + 8; // shared row stride of a warp's P tile

struct Smem {
	half k[2][kBK * kKV]; // double buffered: the next tile is stored while this one is read
	half v[2][kBK * kKV];
	half p[kWarps][16 * kPS];
};

__device__ __forceinline__ float warp_row_max( float x ) {
	x = fmaxf( x, __shfl_xor_sync( 0xffffffffu, x, 2 ) );
	return fmaxf( x, __shfl_xor_sync( 0xffffffffu, x, 8 ) );
}

__device__ __forceinline__ float warp_row_sum( float x ) {
	x += __shfl_xor_sync( 0xffffffffu, x, 2 );
	return x + __shfl_xor_sync( 0xffffffffu, x, 8 );
}

// q: [q_rows, heads, 128] ...; one block = 128 query rows of one head
template <bool Causal, bool Load, bool Store>
__global__ void __launch_bounds__( kThreads, 1 ) h3_attn_d128( const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v, half* __restrict__ out, int q_rows,
	int k_rows, int heads, float scale_log2, float* __restrict__ ostate, float2* __restrict__ mlstate ) {
	extern __shared__ __align__( 16 ) unsigned char smem_raw[];
	Smem& sm = *reinterpret_cast<Smem*>( smem_raw );
	const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
	const size_t stride = static_cast<size_t>( heads ) * kD;
	const int head = blockIdx.y;
	const half* qh = q + static_cast<size_t>( head ) * kD;
	const half* kh = k + static_cast<size_t>( head ) * kD;
	const half* vh = v + static_cast<size_t>( head ) * kD;
	half* oh = out + static_cast<size_t>( head ) * kD;
	const int q0 = blockIdx.x * kBQ;
	const int wrow0 = q0 + warp * 16;

	// Volta accumulator layout of a 16x16 fragment: element i of lane `lane` sits at row = row_base + 2 * ((i >> 1) & 1), col = col_base + (i & 1) + 4 * ((i >> 2) & 1)
	const int row_base = ( lane & 1 ) + ( ( lane >> 2 ) & 1 ) * 8 + ( ( lane >> 4 ) & 1 ) * 4;
	const int col_base = ( ( lane >> 1 ) & 1 ) * 2 + ( ( lane >> 3 ) & 1 ) * 8;

	// Q fragments: staged through shared memory (rows past q_rows read as zeros)
	{
		half* stage = &sm.k[0][0] + warp * 16 * kKV; // both K buffers: 8 warps x 16 rows x 136 halves
		for ( int idx = lane; idx < 16 * 16; idx += 32 ) {
			const int r = idx >> 4, c = idx & 15;
			uint4 val = make_uint4( 0, 0, 0, 0 );
			if ( wrow0 + r < q_rows ) { val = __ldg( reinterpret_cast<const uint4*>( qh + static_cast<size_t>( wrow0 + r ) * stride ) + c ); }
			*reinterpret_cast<uint4*>( stage + r * kKV + c * 8 ) = val;
		}
		__syncwarp();
	}
	wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qa[8];
	#pragma unroll
	for ( int kk = 0; kk < 8; ++kk ) { wmma::load_matrix_sync( qa[kk], &sm.k[0][0] + warp * 16 * kKV + kk * 16, kKV ); }
	__syncthreads(); // the staging area is K buffer storage

	wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[8];
	#pragma unroll
	for ( int dn = 0; dn < 8; ++dn ) { wmma::fill_fragment( o[dn], 0.0f ); }
	float m[2] = { -INFINITY, -INFINITY }, l[2] = { 0.0f, 0.0f };
	// partial passes (mode bit 0: continue from the saved state, bit 1: save the state instead of finishing): the accumulators of earlier key ranges
	if constexpr ( Load ) {
		#pragma unroll
		for ( int r = 0; r < 2; ++r ) {
			const int row = wrow0 + row_base + r * 2;
			if ( row < q_rows ) {
				const float2 ml = mlstate[static_cast<size_t>( row ) * heads + head];
				m[r] = ml.x;
				l[r] = ( lane & 0xa ) == 0 ? ml.y : 0.0f; // l is a per-lane partial sum until the row reduction; one lane of the row group carries the saved total
			}
		}
		#pragma unroll
		for ( int dn = 0; dn < 8; ++dn ) {
			#pragma unroll
			for ( int e = 0; e < 8; e += 2 ) {
				const int row = wrow0 + row_base + ( ( e >> 1 ) & 1 ) * 2, col = dn * 16 + col_base + ( ( e >> 2 ) & 1 ) * 4;
				if ( row < q_rows ) {
					const float2 v2 = *reinterpret_cast<const float2*>( ostate + static_cast<size_t>( row ) * stride + head * kD + col );
					o[dn].x[e] = v2.x;
					o[dn].x[e + 1] = v2.y;
				}
			}
		}
	}
	half* sp = sm.p[warp];

	int ntiles = ( k_rows + kBK - 1 ) / kBK;
	if constexpr ( Causal ) { ntiles = min( ntiles, ( q0 + kBQ + kBK - 1 ) / kBK ); }
	// each thread moves 4 uint4 of K and 4 of V per tile: global -> registers (issued before the math of the previous tile) -> shared (after it)
	uint4 kreg[4], vreg[4];
	auto fetch = [&]( int t ) {
		const int k0 = t * kBK, valid = min( kBK, k_rows - k0 );
		#pragma unroll
		for ( int u = 0; u < 4; ++u ) {
			const int idx = tid + u * kThreads, r = idx >> 4, c = idx & 15;
			kreg[u] = make_uint4( 0, 0, 0, 0 );
			vreg[u] = make_uint4( 0, 0, 0, 0 );
			if ( r < valid ) {
				kreg[u] = __ldg( reinterpret_cast<const uint4*>( kh + static_cast<size_t>( k0 + r ) * stride ) + c );
				vreg[u] = __ldg( reinterpret_cast<const uint4*>( vh + static_cast<size_t>( k0 + r ) * stride ) + c );
			}
		}
	};
	auto stash = [&]( int buf ) {
		#pragma unroll
		for ( int u = 0; u < 4; ++u ) {
			const int idx = tid + u * kThreads, r = idx >> 4, c = idx & 15;
			*reinterpret_cast<uint4*>( &sm.k[buf][r * kKV + c * 8] ) = kreg[u];
			*reinterpret_cast<uint4*>( &sm.v[buf][r * kKV + c * 8] ) = vreg[u];
		}
	};
	fetch( 0 );
	stash( 0 );
	for ( int t = 0; t < ntiles; ++t ) {
		const int k0 = t * kBK, valid = min( kBK, k_rows - k0 );
		const int buf = t & 1;
		__syncthreads(); // tile t is in shared memory; every warp is done with tile t - 1 (buffer buf ^ 1)
		if ( t + 1 < ntiles ) { fetch( t + 1 ); }

		wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[4];
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) { wmma::fill_fragment( s[j], 0.0f ); }
		#pragma unroll
		for ( int kk = 0; kk < 8; ++kk ) {
			#pragma unroll
			for ( int j = 0; j < 4; ++j ) {
				wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kb;
				wmma::load_matrix_sync( kb, &sm.k[buf][j * 16 * kKV + kk * 16], kKV );
				wmma::mma_sync( s[j], qa[kk], kb, s[j] );
			}
		}

		// online softmax on the accumulators (scores in log2 units)
		float tmax[2] = { -INFINITY, -INFINITY };
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				const int col = j * 16 + col_base + ( i & 1 ) + ( ( i >> 2 ) & 1 ) * 4;
				float x = s[j].x[i] * scale_log2;
				bool ok = col < valid;
				if constexpr ( Causal ) { ok = ok && ( k0 + col <= wrow0 + row_base + ( ( i >> 1 ) & 1 ) * 2 ); }
				x = ok ? x : -INFINITY;
				s[j].x[i] = x;
				tmax[( i >> 1 ) & 1] = fmaxf( tmax[( i >> 1 ) & 1], x );
			}
		}
		float factor[2], mnew[2];
		#pragma unroll
		for ( int r = 0; r < 2; ++r ) {
			tmax[r] = warp_row_max( tmax[r] );
			mnew[r] = fmaxf( m[r], tmax[r] );
			factor[r] = mnew[r] == -INFINITY ? 1.0f : exp2f( m[r] - mnew[r] );
		}
		float lsum[2] = { 0.0f, 0.0f };
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) {
				const int r = ( i >> 1 ) & 1;
				const float p = mnew[r] == -INFINITY ? 0.0f : exp2f( s[j].x[i] - mnew[r] );
				lsum[r] += p;
				s[j].x[i] = p;
			}
		}
		#pragma unroll
		for ( int r = 0; r < 2; ++r ) {
			l[r] = l[r] * factor[r] + lsum[r];
			m[r] = mnew[r];
		}
		#pragma unroll
		for ( int dn = 0; dn < 8; ++dn ) {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) { o[dn].x[i] *= factor[( i >> 1 ) & 1]; }
		}
		// P -> shared (this warp's 16 x 64 tile)
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			#pragma unroll
			for ( int e = 0; e < 8; e += 2 ) {
				const int row = row_base + ( ( e >> 1 ) & 1 ) * 2, col = j * 16 + col_base + ( ( e >> 2 ) & 1 ) * 4;
				*reinterpret_cast<half2*>( sp + row * kPS + col ) = __floats2half2_rn( s[j].x[e], s[j].x[e + 1] );
			}
		}
		__syncwarp();
		wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa[4];
		#pragma unroll
		for ( int kk = 0; kk < 4; ++kk ) { wmma::load_matrix_sync( pa[kk], sp + kk * 16, kPS ); }
		#pragma unroll
		for ( int dn = 0; dn < 8; ++dn ) {
			#pragma unroll
			for ( int kk = 0; kk < 4; ++kk ) {
				wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb;
				wmma::load_matrix_sync( vb, &sm.v[buf][kk * 16 * kKV + dn * 16], kKV );
				wmma::mma_sync( o[dn], pa[kk], vb, o[dn] );
			}
		}
		if ( t + 1 < ntiles ) { stash( buf ^ 1 ); }
	}

	if constexpr ( Store ) {
		#pragma unroll
		for ( int dn = 0; dn < 8; ++dn ) {
			#pragma unroll
			for ( int e = 0; e < 8; e += 2 ) {
				const int row = wrow0 + row_base + ( ( e >> 1 ) & 1 ) * 2, col = dn * 16 + col_base + ( ( e >> 2 ) & 1 ) * 4;
				if ( row < q_rows ) { *reinterpret_cast<float2*>( ostate + static_cast<size_t>( row ) * stride + head * kD + col ) = make_float2( o[dn].x[e], o[dn].x[e + 1] ); }
			}
		}
		#pragma unroll
		for ( int r = 0; r < 2; ++r ) {
			const float ls = warp_row_sum( l[r] );
			const int row = wrow0 + row_base + r * 2;
			if ( ( lane & 0xa ) == 0 && row < q_rows ) { mlstate[static_cast<size_t>( row ) * heads + head] = make_float2( m[r], ls ); }
		}
		return;
	}
	// epilogue: O / l
	float inv[2];
	#pragma unroll
	for ( int r = 0; r < 2; ++r ) {
		const float ls = warp_row_sum( l[r] );
		inv[r] = ls > 0.0f ? 1.0f / ls : 0.0f;
	}
	#pragma unroll
	for ( int dn = 0; dn < 8; ++dn ) {
		#pragma unroll
		for ( int e = 0; e < 8; e += 2 ) {
			const int r = ( e >> 1 ) & 1;
			const int row = wrow0 + row_base + r * 2, col = dn * 16 + col_base + ( ( e >> 2 ) & 1 ) * 4;
			if ( row < q_rows ) {
				*reinterpret_cast<half2*>( oh + static_cast<size_t>( row ) * stride + col ) = __floats2half2_rn( o[dn].x[e] * inv[r], o[dn].x[e + 1] * inv[r] );
			}
		}
	}
}

} // namespace

// the shared-memory opt-in is per device (the DiT runs one thread per GPU)
template <bool Causal, bool Load, bool Store>
static void configure() {
	static std::mutex mu;
	static std::bitset<64> configured;
	int dev = 0;
	CUDA_CHECK( cudaGetDevice( &dev ) );
	std::lock_guard<std::mutex> lock( mu );
	if ( !configured.test( dev & 63 ) ) {
		CUDA_CHECK( cudaFuncSetAttribute( h3_attn_d128<Causal, Load, Store>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( sizeof( Smem ) ) ) );
		configured.set( dev & 63 );
	}
}

bool h3_attention( const half* q, const half* k, const half* v, half* out, int q_rows, int k_rows, int heads, float scale, bool causal, cudaStream_t stream ) {
	if ( q_rows <= 0 ) { return true; }
	if ( k_rows <= 0 || heads <= 0 ) { return false; }
	const size_t smem = sizeof( Smem );
	const dim3 grid( ( q_rows + kBQ - 1 ) / kBQ, heads );
	const float scale_log2 = scale * 1.4426950408889634f;
	if ( causal ) {
		configure<true, false, false>();
		h3_attn_d128<true, false, false><<<grid, kThreads, smem, stream>>>( q, k, v, out, q_rows, k_rows, heads, scale_log2, nullptr, nullptr );
	} else {
		configure<false, false, false>();
		h3_attn_d128<false, false, false><<<grid, kThreads, smem, stream>>>( q, k, v, out, q_rows, k_rows, heads, scale_log2, nullptr, nullptr );
	}
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

bool h3_attention_partial( const half* q, const half* k, const half* v, half* out, float* state_o, float2* state_ml, int q_rows, int k_rows, int heads, float scale, bool first, bool last,
	cudaStream_t stream ) {
	if ( q_rows <= 0 ) { return true; }
	if ( k_rows <= 0 || heads <= 0 || !state_o || !state_ml || ( last && !out ) ) { return false; }
	const size_t smem = sizeof( Smem );
	const dim3 grid( ( q_rows + kBQ - 1 ) / kBQ, heads );
	const float sl = scale * 1.4426950408889634f;
	if ( first && last ) {
		configure<false, false, false>();
		h3_attn_d128<false, false, false><<<grid, kThreads, smem, stream>>>( q, k, v, out, q_rows, k_rows, heads, sl, nullptr, nullptr );
	} else if ( first ) {
		configure<false, false, true>();
		h3_attn_d128<false, false, true><<<grid, kThreads, smem, stream>>>( q, k, v, out, q_rows, k_rows, heads, sl, state_o, state_ml );
	} else if ( last ) {
		configure<false, true, false>();
		h3_attn_d128<false, true, false><<<grid, kThreads, smem, stream>>>( q, k, v, out, q_rows, k_rows, heads, sl, state_o, state_ml );
	} else {
		configure<false, true, true>();
		h3_attn_d128<false, true, true><<<grid, kThreads, smem, stream>>>( q, k, v, out, q_rows, k_rows, heads, sl, state_o, state_ml );
	}
	CUDA_CHECK( cudaGetLastError() );
	return true;
}

} // namespace engine::video
