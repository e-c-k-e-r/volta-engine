#pragma once

// The EXL3 trellis encoder: a Viterbi search over the tail-biting ring of one 16x16 tile (256 weights in stream order), ported from
// exllamav3's quantize_tiles_kernel.cuh and quantize_tiles_frac_kernel.cuh (MIT, see LICENSE). The decode is the inference codebook
// (../codebook.cuh), so what the encoder chooses is exactly what the kernels read back. Each step evaluates all 65536 window values as
// 2^(16-K) out-states times 2^K candidates; the first pass starts free from the middle of the ring to find where the ring closes, the second
// pass is pinned to that state and writes the codes.

#include <cuda_fp16.h>
#include <cstdint>
#include "../codebook.cuh"

#ifndef EXL3E_H_INF
#define EXL3E_H_INF __ushort_as_half( 0x7c00 )
#endif

// Smallest K whose two cost arrays (2 * (65536 >> K) halves) live in shared memory; below it they are in the caller's global scratch
#define EXL3E_SHARED_COSTS_MIN_K 2

// Cost arrays (of the ping-pong pair) in shared memory where both do not fit or would cost occupancy: K = 1, and the half-integer rates by
// their KA (0, 1: one, the other in global scratch, so every step has one global stream instead of two; 2: both)
__host__ __device__ constexpr int exl3e_k1_shared_costs() { return 1; }
__host__ __device__ constexpr int exl3e_half_shared_costs( int KA ) { return KA == 1 ? 1 : KA == 2 ? 1 : 2; }

// Threads per block: each step has 32768 >> K pairs of out-states, so from K = 7 on a 512-thread block would leave most threads idle; K = 1's
// shared cost array allows one block per SM, so it takes 1024 threads to keep the SM's warps
__host__ __device__ constexpr int exl3e_num_threads( int K ) { return K >= 7 ? ( 32768 >> K ) : K == 1 ? 1024 : 512; }
__host__ __device__ constexpr int exl3e_min_blocks( int K ) { return K == 6 ? 1 : 1024 / exl3e_num_threads( K ); }

// Shared memory of the integer-rate kernel: staged tile, cost scales, argmin scratch, and from K = 2 on both cost arrays (K = 1:
// exl3e_k1_shared_costs of them)
__host__ __device__ constexpr int exl3e_shmem( int K ) {
	return ( K >= EXL3E_SHARED_COSTS_MIN_K ? 2 : exl3e_k1_shared_costs() ) * ( 65536 >> K ) * 2 + 2 * 256 * 2 + 64 + 128;
}
// The half-integer rates' kernel: staged tile, cost scales, argmin scratch and exl3e_half_shared_costs( KA ) cost arrays of 65536 >> KA halves
__host__ __device__ constexpr int exl3e_half_shmem( int KA ) { return 2 * 256 * 2 + 32 * 4 + 128 + exl3e_half_shared_costs( KA ) * ( 65536 >> KA ) * 2; }

// Weighted metric (per-position weights w, the cost sum w_i e_i^2): the stored costs are the running costs divided by the weight of their position,
// so a step stays e^2 plus an in-cost, and the costs it stores are scaled once by w(ri) / w(ri + 1), the same factor for every state (the argmin
// and the pinned pass are unchanged). scale[ri] for the 256 ring positions; 1 everywhere without weights
__device__ __forceinline__ void exl3e_load_scales( const float* weights, half* scale, int nt ) {
	for ( int i = threadIdx.x; i < 256; i += nt ) {
		float s = 1.0f;
		if ( weights ) {
			const float a = weights[i], b = weights[( i + 1 ) & 255];
			s = a > 0.0f && b > 0.0f ? a / b : 1.0f;
		}
		scale[i] = __float2half_rn( s );
	}
}

__device__ __forceinline__ uint32_t exl3e_warp_min( uint32_t value ) {
	#pragma unroll
	for ( int offset = 16; offset; offset >>= 1 ) value = min( value, __shfl_xor_sync( 0xffffffff, value, offset ) );
	return value;
}

// ri = (i + roll) mod 256, with i < 256 and roll in {0, 128}
__device__ __forceinline__ int exl3e_ring( int i, int roll ) {
	int ri = i + roll;
	if ( ri >= 256 ) ri -= 256;
	return ri;
}

// The out-state with the lowest final cost. The rank makes the tie order independent of the block size: bit-reversed warp, then bit-reversed
// lane, then increasing 1024-state stripe. Every cost is a nonnegative half, so packed unsigned minima order them. All costs inf or NaN
// (degenerate input): a valid state, so the traceback stays in bounds
template <int NT>
__device__ __forceinline__ int exl3e_argmin( const half* costs, int count, int* sh_idx ) {
	constexpr int NW = NT / 32;
	const int thread = threadIdx.x;
	uint32_t best = 0x7c00ffffu;
	for ( int e = thread; e < count; e += NT ) {
		const unsigned v = e & 1023;
		const unsigned rank = ( ( __brev( v >> 5 ) >> 27 ) << 10 ) | ( ( __brev( v & 31 ) >> 27 ) << 5 ) | ( e >> 10 );
		const unsigned key = ( (uint32_t) __half_as_ushort( costs[e] ) << 16 ) | rank;
		best = min( best, key );
	}
	best = exl3e_warp_min( best );
	if ( ( thread & 31 ) == 0 ) ( (uint32_t*) sh_idx )[thread >> 5] = best;
	__syncthreads();
	if ( thread < 32 ) {
		best = thread < NW ? ( (uint32_t*) sh_idx )[thread] : 0x7c00ffffu;
		best = exl3e_warp_min( best );
	}
	const unsigned rank = best & 65535;
	const unsigned v = ( ( __brev( rank >> 10 ) >> 27 ) << 5 ) | ( __brev( ( rank >> 5 ) & 31 ) >> 27 );
	return best >= 0x7c000000u ? 0 : (int) ( ( ( rank & 31 ) << 10 ) | v );
}

// Two candidate decodes of one step for out-states out_edge_idx and out_edge_idx + 1, candidate k (k = 0 the in-state out_edge_idx >> K).
// mcg and mul1 are a multiply modulo 2^32 of the window, so successive candidates step the products by a constant
template <int K, int cb>
struct Exl3eDecoder {
	uint32_t p0, p1;
	__device__ __forceinline__ half2 first( int out_edge_idx ) {
		if constexpr ( cb == 1 ) {
			p0 = (uint32_t) out_edge_idx * 0xCBAC1FEDu;
			p1 = p0 + 0xCBAC1FEDu;
			return decode_mcg_product_2( p0, p1 );
		} else if constexpr ( cb == 2 ) {
			p0 = (uint32_t) out_edge_idx * 0x83DCD12Du;
			p1 = p0 + 0x83DCD12Du;
			return decode_mul1_product_2( p0, p1 );
		} else {
			return decode_3inst_2<cb>( out_edge_idx, out_edge_idx + 1 );
		}
	}
	__device__ __forceinline__ half2 next( int state0 ) {
		constexpr int Kr = 16 - K;
		if constexpr ( cb == 1 ) {
			constexpr uint32_t step = 0xCBAC1FEDu << Kr;
			p0 += step;
			p1 += step;
			return decode_mcg_product_2( p0, p1 );
		} else if constexpr ( cb == 2 ) {
			constexpr uint32_t step = 0x83DCD12Du << Kr;
			p0 += step;
			p1 += step;
			return decode_mul1_product_2( p0, p1 );
		} else {
			return decode_3inst_2<cb>( state0, state0 + 1 );
		}
	}
};

// The history of a step: per out-state the chosen candidate k, the in-state being ( ( k << ( 16 - KOUT ) ) | out ) >> KIN. Up to K = 4 bits
// (k < 16) a pair of out-states shares a byte (the even one's k low), else a 16-bit word holds the in-state itself: the low rates' 32768
// out-states per step make the history the encoder's main memory traffic
__host__ __device__ constexpr bool exl3e_packed_hist( int kmax ) { return kmax <= 4; }

// The in-state of out-state `edge` at a step from its history row (KIN, KOUT the step's widths; packed: bytes, else 16-bit in-states)
template <bool PACKED>
__device__ __forceinline__ int exl3e_hist_in( const void* hist_row, int edge, int kin, int kout ) {
	if constexpr ( PACKED ) {
		const uint8_t b = static_cast<const uint8_t*>( hist_row )[edge >> 1];
		const int k = ( edge & 1 ) ? b >> 4 : b & 15;
		return ( ( k << ( 16 - kout ) ) | edge ) >> kin;
	} else {
		return static_cast<const uint16_t*>( hist_row )[edge];
	}
}

// One Viterbi step at position ri: out-state costs from the in-state costs (first: no in-costs; with pre_state >= 0 only that in-state is
// allowed), and the chosen candidate of every out-state into the history row of ri (exl3e_packed_hist)
template <int K, int cb, int NT>
__device__ __forceinline__ void exl3e_step( half w, const half* costs_in, half* costs_out, void* hist_row, bool first, int pre_state,
	bool write_hist, bool weighted, half scale ) {
	const half2 scale2 = __half2half2( scale );
	constexpr int Kr = 16 - K;
	constexpr int max_q = 1 << K;
	constexpr int edges = 65536 >> K;
	const half2 w2 = __half2half2( w );
	const half2 inf2 = __half2half2( EXL3E_H_INF );
	for ( int out_edge_idx = 2 * threadIdx.x; out_edge_idx < edges; out_edge_idx += 2 * NT ) {
		Exl3eDecoder<K, cb> dec;
		int in_edge_idx = out_edge_idx >> K;
		half2 dh2 = __hsub2( dec.first( out_edge_idx ), w2 );
		half2 min_err2;
		if ( first ) {
			min_err2 = __hmul2( dh2, dh2 );
			if ( pre_state >= 0 && in_edge_idx != pre_state ) min_err2 = inf2;
		} else {
			min_err2 = __hfma2( dh2, dh2, __half2half2( costs_in[in_edge_idx] ) );
		}
		int min_in_edge0 = in_edge_idx;
		int min_in_edge1 = in_edge_idx;
		int min_k0 = 0, min_k1 = 0;
		#pragma unroll
		for ( int k = 1; k < max_q; ++k ) {
			const int state0 = ( k << Kr ) | out_edge_idx;
			in_edge_idx = state0 >> K;
			dh2 = __hsub2( dec.next( state0 ), w2 );
			half2 err2;
			if ( first ) {
				err2 = __hmul2( dh2, dh2 );
				if ( pre_state >= 0 && in_edge_idx != pre_state ) err2 = inf2;
			} else {
				err2 = __hfma2( dh2, dh2, __half2half2( costs_in[in_edge_idx] ) );
			}
			if ( __hlt( __low2half( err2 ), __low2half( min_err2 ) ) ) {
				min_err2 = __halves2half2( __low2half( err2 ), __high2half( min_err2 ) );
				min_in_edge0 = in_edge_idx;
				min_k0 = k;
			}
			if ( __hlt( __high2half( err2 ), __high2half( min_err2 ) ) ) {
				min_err2 = __halves2half2( __low2half( min_err2 ), __high2half( err2 ) );
				min_in_edge1 = in_edge_idx;
				min_k1 = k;
			}
		}
		reinterpret_cast<half2*>( costs_out )[out_edge_idx >> 1] = weighted ? __hmul2( min_err2, scale2 ) : min_err2;
		if ( write_hist ) {
			if constexpr ( exl3e_packed_hist( K ) ) {
				static_cast<uint8_t*>( hist_row )[out_edge_idx >> 1] = (uint8_t) ( min_k0 | ( min_k1 << 4 ) );
			} else {
				static_cast<uint16_t*>( hist_row )[out_edge_idx] = (uint16_t) min_in_edge0;
				static_cast<uint16_t*>( hist_row )[out_edge_idx + 1] = (uint16_t) min_in_edge1;
			}
		}
	}
}

// Integer rate K (1..8), codebook cb (0 3INST, 1 mcg, 2 mul1). One block per tile; temp_costs (K < 2) holds 2 * (65536 >> K) halves per tile
// of the batch, temp_edges 256 * (65536 >> K) per tile; weights (256 per tile, stream order, or null: plain squared error) weight the metric.
// open: no ring, one pass from position 0 with a free start state (not packable: the first positions' windows hold bits from before the tile)
template <int K, int cb>
__global__ __launch_bounds__( exl3e_num_threads( K ), exl3e_min_blocks( K ) )
void exl3_encode_tiles_kernel(
	const float* __restrict__ input_tiles,
	const float* __restrict__ weights,
	float* __restrict__ output_tiles,
	uint16_t* __restrict__ output_indices,
	half* __restrict__ temp_costs_ptr,
	uint16_t* __restrict__ temp_edges_ptr,
	int open
) {
	constexpr int NT = exl3e_num_threads( K );
	constexpr int edges = 65536 >> K;
	constexpr bool PK = exl3e_packed_hist( K );
	constexpr size_t row_bytes = PK ? edges / 2 : edges * sizeof( uint16_t );   // a step's history
	extern __shared__ uint8_t shbuf[];
	uint8_t* sh = shbuf;

	const int tile_idx = blockIdx.x;
	const int thread = threadIdx.x;
	const float* input_tile = input_tiles + 256 * tile_idx;
	float* output_tile = output_tiles + 256 * tile_idx;
	uint16_t* out_idx = output_indices + 256 * tile_idx;
	uint8_t* hist = reinterpret_cast<uint8_t*>( temp_edges_ptr ) + 256 * row_bytes * tile_idx;

	half* sh_input_tile = (half*) sh;
	sh += 256 * sizeof( half );
	half* sh_scale = (half*) sh;
	sh += 256 * sizeof( half );
	sh += 32 * sizeof( half );
	int* sh_idx = (int*) sh;
	sh += 32 * sizeof( int );
	constexpr int SHC = K >= EXL3E_SHARED_COSTS_MIN_K ? 2 : exl3e_k1_shared_costs();   // cost arrays in shared memory
	half* g_costs = temp_costs_ptr + (size_t) 2 * edges * tile_idx;
	half* temp_costs = SHC >= 1 ? (half*) sh : g_costs;
	half* temp_costs_inc = SHC >= 2 ? (half*) sh + edges : g_costs + edges;
	const bool weighted = weights != nullptr;

	for ( int i = thread; i < 256; i += NT ) sh_input_tile[i] = __float2half_rn( input_tile[i] );
	exl3e_load_scales( weighted ? weights + 256 * tile_idx : nullptr, sh_scale, NT );
	__syncthreads();

	auto forward = [&]( int roll, int pre_state ) {
		for ( int i = 0; i < 256; ++i ) {
			const int ri = exl3e_ring( i, roll );
			half* t = temp_costs;
			temp_costs = temp_costs_inc;
			temp_costs_inc = t;
			// The first pass only traces back to position 0, so its history for the rolled second half (ri >= 128) is never read
			exl3e_step<K, cb, NT>( sh_input_tile[ri], temp_costs_inc, temp_costs, hist + row_bytes * ri, i == 0, pre_state,
				pre_state >= 0 || ri < 128 || open, weighted, sh_scale[ri] );
			__syncthreads();
		}
	};

	auto backward = [&]( int roll, bool write, int edge ) {
		if ( thread == 0 ) {
			for ( int i = 255; i >= 0; --i ) {
				const int ri = exl3e_ring( i, roll );
				const int prev_edge = exl3e_hist_in<PK>( hist + row_bytes * ri, edge, K, K );
				const int encoded = ( prev_edge << K ) | edge;
				edge = prev_edge;
				if ( write ) {
					out_idx[ri] = (uint16_t) encoded;
					output_tile[ri] = __half2float( decode_3inst<cb>( encoded ) );
				} else if ( ri == 0 ) break;
			}
			sh_idx[0] = edge;
		}
		__syncthreads();
		return sh_idx[0];
	};

	if ( open ) {
		forward( 0, -1 );
		backward( 0, true, exl3e_argmin<NT>( temp_costs, edges, sh_idx ) );
		return;
	}
	forward( 128, -1 );
	const int end_state = backward( 128, false, exl3e_argmin<NT>( temp_costs, edges, sh_idx ) );
	forward( 0, end_state );
	backward( 0, true, end_state );
}

// Half-integer rate KA + 0.5 (mul1 only): the window advances by D(i) = KA + bit (i mod 16) of MASK bits at position i (MASK 0xAAAA alternates
// KA and KA + 1). The node between positions i - 1 and i is the (16 - D(i))-bit overlap of their windows. Costs and history in global scratch,
// sized for the wider node space (65536 >> KA). Threads per block: KA = 1's shared cost array allows one block per SM, so it takes 1024 threads
// to keep the SM's warps; the others 512, two blocks
__host__ __device__ constexpr int exl3e_half_threads( int KA ) { return KA == 1 ? 1024 : 512; }

template <int KIN, int KOUT, int NT>
__device__ __forceinline__ void exl3e_half_step( half w, const half* costs_in, half* costs_out, uint8_t* hist_row, bool first, int pre_state,
	bool write_hist, bool weighted, half scale ) {
	static_assert( exl3e_packed_hist( KOUT ), "exl3e_half_step: the half-integer rates' candidates fit the packed history" );
	const half2 scale2 = __half2half2( scale );
	constexpr int Kr = 16 - KOUT;
	constexpr int max_q = 1 << KOUT;
	constexpr int edges_out = 65536 >> KOUT;
	const half2 w2 = __half2half2( w );
	const half2 inf2 = __half2half2( EXL3E_H_INF );
	for ( int out_edge_idx = 2 * threadIdx.x; out_edge_idx < edges_out; out_edge_idx += 2 * NT ) {
		int in_edge_idx = out_edge_idx >> KIN;
		uint32_t p0 = (uint32_t) out_edge_idx * 0x83DCD12Du;
		uint32_t p1 = p0 + 0x83DCD12Du;
		half2 dh2 = __hsub2( decode_mul1_product_2( p0, p1 ), w2 );
		half2 min_err2;
		if ( first ) {
			min_err2 = __hmul2( dh2, dh2 );
			if ( pre_state >= 0 && in_edge_idx != pre_state ) min_err2 = inf2;
		} else {
			min_err2 = __hfma2( dh2, dh2, __half2half2( costs_in[in_edge_idx] ) );
		}
		int min_k0 = 0, min_k1 = 0;
		#pragma unroll
		for ( int k = 1; k < max_q; ++k ) {
			const int state0 = ( k << Kr ) | out_edge_idx;
			in_edge_idx = state0 >> KIN;
			constexpr uint32_t step = 0x83DCD12Du << Kr;
			p0 += step;
			p1 += step;
			dh2 = __hsub2( decode_mul1_product_2( p0, p1 ), w2 );
			half2 err2;
			if ( first ) {
				err2 = __hmul2( dh2, dh2 );
				if ( pre_state >= 0 && in_edge_idx != pre_state ) err2 = inf2;
			} else {
				err2 = __hfma2( dh2, dh2, __half2half2( costs_in[in_edge_idx] ) );
			}
			if ( __hlt( __low2half( err2 ), __low2half( min_err2 ) ) ) {
				min_err2 = __halves2half2( __low2half( err2 ), __high2half( min_err2 ) );
				min_k0 = k;
			}
			if ( __hlt( __high2half( err2 ), __high2half( min_err2 ) ) ) {
				min_err2 = __halves2half2( __low2half( min_err2 ), __high2half( err2 ) );
				min_k1 = k;
			}
		}
		reinterpret_cast<half2*>( costs_out )[out_edge_idx >> 1] = weighted ? __hmul2( min_err2, scale2 ) : min_err2;
		if ( write_hist ) {
			hist_row[out_edge_idx >> 1] = (uint8_t) ( min_k0 | ( min_k1 << 4 ) );
		}
	}
}

template <int KA, uint32_t MASK>
__global__ __launch_bounds__( exl3e_half_threads( KA ), KA == 1 ? 1 : 2 )
void exl3_encode_tiles_half_kernel(
	const float* __restrict__ input_tiles,
	const float* __restrict__ weights,
	float* __restrict__ output_tiles,
	uint16_t* __restrict__ output_indices,
	half* __restrict__ temp_costs_ptr,
	uint16_t* __restrict__ temp_edges_ptr,
	int open
) {
	constexpr int NT = exl3e_half_threads( KA );
	constexpr int KB = KA + 1;
	constexpr int edges_max = 65536 >> KA;
	// Both passes end on position 255 or 127, whose out-width is set by D(0) (the ring wraps)
	constexpr int edges_last = 65536 >> ( KA + ( MASK & 1 ) );
	extern __shared__ uint8_t shbuf[];
	uint8_t* sh = shbuf;

	const int tile_idx = blockIdx.x;
	const int thread = threadIdx.x;
	const float* input_tile = input_tiles + 256 * tile_idx;
	float* output_tile = output_tiles + 256 * tile_idx;
	uint16_t* out_idx = output_indices + 256 * tile_idx;
	constexpr size_t row_bytes = edges_max / 2;   // a step's packed history
	uint8_t* hist = reinterpret_cast<uint8_t*>( temp_edges_ptr ) + 256 * row_bytes * tile_idx;

	half* sh_input_tile = (half*) sh;
	sh += 256 * sizeof( half );
	half* sh_scale = (half*) sh;
	sh += 256 * sizeof( half );
	int* sh_idx = (int*) sh;
	sh += 32 * sizeof( int ) + 128;

	constexpr int SHC = exl3e_half_shared_costs( KA );   // cost arrays in shared memory
	half* g_costs = temp_costs_ptr + (size_t) 2 * edges_max * tile_idx;
	half* temp_costs = SHC >= 1 ? (half*) sh : g_costs;
	half* temp_costs_inc = SHC >= 2 ? (half*) sh + edges_max : g_costs + edges_max;
	const bool weighted = weights != nullptr;

	for ( int i = thread; i < 256; i += NT ) sh_input_tile[i] = __float2half_rn( input_tile[i] );
	exl3e_load_scales( weighted ? weights + 256 * tile_idx : nullptr, sh_scale, NT );
	__syncthreads();

	auto forward = [&]( int roll, int pre_state ) {
		for ( int i = 0; i < 256; ++i ) {
			const int ri = exl3e_ring( i, roll );
			half* t = temp_costs;
			temp_costs = temp_costs_inc;
			temp_costs_inc = t;
			const bool first = i == 0;
			const bool write_hist = pre_state >= 0 || ri < 128 || open;
			uint8_t* hist_row = hist + row_bytes * ri;
			const half w = sh_input_tile[ri];
			const half sc = sh_scale[ri];
			// KIN = D(ri), KOUT = D(ri + 1)
			const int din = ( MASK >> ( ri & 15 ) ) & 1, dout = ( MASK >> ( ( ri + 1 ) & 15 ) ) & 1;
			switch ( din * 2 + dout ) {
				case 0: {
					exl3e_half_step<KA, KA, NT>( w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, weighted, sc );
				} break;
				case 1: {
					exl3e_half_step<KA, KB, NT>( w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, weighted, sc );
				} break;
				case 2: {
					exl3e_half_step<KB, KA, NT>( w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, weighted, sc );
				} break;
				default: {
					exl3e_half_step<KB, KB, NT>( w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist, weighted, sc );
				} break;
			}
			__syncthreads();
		}
	};

	auto backward = [&]( int roll, bool write, int edge ) {
		if ( thread == 0 ) {
			for ( int i = 255; i >= 0; --i ) {
				const int ri = exl3e_ring( i, roll );
				const int kin = KA + ( ( MASK >> ( ri & 15 ) ) & 1 ), kout = KA + ( ( MASK >> ( ( ri + 1 ) & 15 ) ) & 1 );
				const int prev_edge = exl3e_hist_in<true>( hist + row_bytes * ri, edge, kin, kout );
				const int encoded = ( ( prev_edge << kin ) | edge ) & 0xFFFF;
				edge = prev_edge;
				if ( write ) {
					out_idx[ri] = (uint16_t) encoded;
					output_tile[ri] = __half2float( decode_3inst<2>( encoded ) );
				} else if ( ri == 0 ) break;
			}
			sh_idx[0] = edge;
		}
		__syncthreads();
		return sh_idx[0];
	};

	if ( open ) {
		forward( 0, -1 );
		backward( 0, true, exl3e_argmin<NT>( temp_costs, edges_last, sh_idx ) );
		return;
	}
	forward( 128, -1 );
	const int end_state = backward( 128, false, exl3e_argmin<NT>( temp_costs, edges_last, sh_idx ) );
	forward( 0, end_state );
	backward( 0, true, end_state );
}
