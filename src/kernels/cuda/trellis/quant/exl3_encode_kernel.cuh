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

// Threads per block: each step has 32768 >> K pairs of out-states, so from K = 7 on a 512-thread block would leave most threads idle
__host__ __device__ constexpr int exl3e_num_threads( int K ) { return K >= 7 ? ( 32768 >> K ) : 512; }
__host__ __device__ constexpr int exl3e_min_blocks( int K ) { return K == 6 ? 1 : 1024 / exl3e_num_threads( K ); }

// Shared memory of the integer-rate kernel: staged tile, argmin scratch, and from K = 2 on both cost arrays
__host__ __device__ constexpr int exl3e_shmem( int K ) {
	return ( K >= EXL3E_SHARED_COSTS_MIN_K ? 2 * ( 65536 >> K ) * 2 : 0 ) + 256 * 2 + 64 + 128;
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

// One Viterbi step at position ri: out-state costs from the in-state costs (first: no in-costs; with pre_state >= 0 only that in-state is
// allowed), and the chosen in-state of every out-state into the history row of ri
template <int K, int cb, int NT>
__device__ __forceinline__ void exl3e_step( half w, const half* costs_in, half* costs_out, uint16_t* hist_row, bool first, int pre_state,
	bool write_hist ) {
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
			}
			if ( __hlt( __high2half( err2 ), __high2half( min_err2 ) ) ) {
				min_err2 = __halves2half2( __low2half( min_err2 ), __high2half( err2 ) );
				min_in_edge1 = in_edge_idx;
			}
		}
		reinterpret_cast<half2*>( costs_out )[out_edge_idx >> 1] = min_err2;
		if ( write_hist ) {
			hist_row[out_edge_idx] = (uint16_t) min_in_edge0;
			hist_row[out_edge_idx + 1] = (uint16_t) min_in_edge1;
		}
	}
}

// Integer rate K (1..8), codebook cb (0 3INST, 1 mcg, 2 mul1). One block per tile; temp_costs (K < 2) holds 2 * (65536 >> K) halves per tile
// of the batch, temp_edges 256 * (65536 >> K) per tile
template <int K, int cb>
__global__ __launch_bounds__( exl3e_num_threads( K ), exl3e_min_blocks( K ) )
void exl3_encode_tiles_kernel(
	const float* __restrict__ input_tiles,
	float* __restrict__ output_tiles,
	uint16_t* __restrict__ output_indices,
	half* __restrict__ temp_costs_ptr,
	uint16_t* __restrict__ temp_edges_ptr
) {
	constexpr int NT = exl3e_num_threads( K );
	constexpr int edges = 65536 >> K;
	extern __shared__ uint8_t shbuf[];
	uint8_t* sh = shbuf;

	const int tile_idx = blockIdx.x;
	const int thread = threadIdx.x;
	const float* input_tile = input_tiles + 256 * tile_idx;
	float* output_tile = output_tiles + 256 * tile_idx;
	uint16_t* out_idx = output_indices + 256 * tile_idx;
	uint16_t* temp_edges = temp_edges_ptr + (size_t) 256 * edges * tile_idx;

	half* sh_input_tile = (half*) sh;
	sh += 256 * sizeof( half );
	sh += 32 * sizeof( half );
	int* sh_idx = (int*) sh;
	sh += 32 * sizeof( int );
	half* temp_costs = K >= EXL3E_SHARED_COSTS_MIN_K ? (half*) sh : temp_costs_ptr + (size_t) 2 * edges * tile_idx;
	half* temp_costs_inc = temp_costs + edges;

	for ( int i = thread; i < 256; i += NT ) sh_input_tile[i] = __float2half_rn( input_tile[i] );
	__syncthreads();

	auto forward = [&]( int roll, int pre_state ) {
		for ( int i = 0; i < 256; ++i ) {
			const int ri = exl3e_ring( i, roll );
			half* t = temp_costs;
			temp_costs = temp_costs_inc;
			temp_costs_inc = t;
			// The first pass only traces back to position 0, so its history for the rolled second half (ri >= 128) is never read
			exl3e_step<K, cb, NT>( sh_input_tile[ri], temp_costs_inc, temp_costs, temp_edges + (size_t) edges * ri, i == 0, pre_state,
				pre_state >= 0 || ri < 128 );
			__syncthreads();
		}
	};

	auto backward = [&]( int roll, bool write, int edge ) {
		if ( thread == 0 ) {
			for ( int i = 255; i >= 0; --i ) {
				const int ri = exl3e_ring( i, roll );
				const int prev_edge = (int) temp_edges[(size_t) edges * ri + edge];
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

	forward( 128, -1 );
	const int end_state = backward( 128, false, exl3e_argmin<NT>( temp_costs, edges, sh_idx ) );
	forward( 0, end_state );
	backward( 0, true, end_state );
}

// Half-integer rate KA + 0.5 (mul1 only): the window advances by D(i) = KA + bit (i mod 16) of MASK bits at position i (MASK 0xAAAA alternates
// KA and KA + 1). The node between positions i - 1 and i is the (16 - D(i))-bit overlap of their windows. Costs and history in global scratch,
// sized for the wider node space (65536 >> KA)
#define EXL3E_HALF_NT 512

template <int KIN, int KOUT>
__device__ __forceinline__ void exl3e_half_step( half w, const half* costs_in, half* costs_out, uint16_t* hist_row, bool first, int pre_state,
	bool write_hist ) {
	constexpr int Kr = 16 - KOUT;
	constexpr int max_q = 1 << KOUT;
	constexpr int edges_out = 65536 >> KOUT;
	const half2 w2 = __half2half2( w );
	const half2 inf2 = __half2half2( EXL3E_H_INF );
	for ( int out_edge_idx = 2 * threadIdx.x; out_edge_idx < edges_out; out_edge_idx += 2 * EXL3E_HALF_NT ) {
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
		int min_in_edge0 = in_edge_idx;
		int min_in_edge1 = in_edge_idx;
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
				min_in_edge0 = in_edge_idx;
			}
			if ( __hlt( __high2half( err2 ), __high2half( min_err2 ) ) ) {
				min_err2 = __halves2half2( __low2half( min_err2 ), __high2half( err2 ) );
				min_in_edge1 = in_edge_idx;
			}
		}
		reinterpret_cast<half2*>( costs_out )[out_edge_idx >> 1] = min_err2;
		if ( write_hist ) {
			hist_row[out_edge_idx] = (uint16_t) min_in_edge0;
			hist_row[out_edge_idx + 1] = (uint16_t) min_in_edge1;
		}
	}
}

template <int KA, uint32_t MASK>
__global__ __launch_bounds__( EXL3E_HALF_NT, 2 )
void exl3_encode_tiles_half_kernel(
	const float* __restrict__ input_tiles,
	float* __restrict__ output_tiles,
	uint16_t* __restrict__ output_indices,
	half* __restrict__ temp_costs_ptr,
	uint16_t* __restrict__ temp_edges_ptr
) {
	constexpr int NT = EXL3E_HALF_NT;
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
	uint16_t* temp_edges = temp_edges_ptr + (size_t) 256 * edges_max * tile_idx;

	half* sh_input_tile = (half*) sh;
	sh += 256 * sizeof( half );
	int* sh_idx = (int*) sh;

	half* temp_costs = temp_costs_ptr + (size_t) 2 * edges_max * tile_idx;
	half* temp_costs_inc = temp_costs + edges_max;

	for ( int i = thread; i < 256; i += NT ) sh_input_tile[i] = __float2half_rn( input_tile[i] );
	__syncthreads();

	auto forward = [&]( int roll, int pre_state ) {
		for ( int i = 0; i < 256; ++i ) {
			const int ri = exl3e_ring( i, roll );
			half* t = temp_costs;
			temp_costs = temp_costs_inc;
			temp_costs_inc = t;
			const bool first = i == 0;
			const bool write_hist = pre_state >= 0 || ri < 128;
			uint16_t* hist_row = temp_edges + (size_t) edges_max * ri;
			const half w = sh_input_tile[ri];
			// KIN = D(ri), KOUT = D(ri + 1)
			const int din = ( MASK >> ( ri & 15 ) ) & 1, dout = ( MASK >> ( ( ri + 1 ) & 15 ) ) & 1;
			switch ( din * 2 + dout ) {
				case 0: {
					exl3e_half_step<KA, KA>( w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist );
				} break;
				case 1: {
					exl3e_half_step<KA, KB>( w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist );
				} break;
				case 2: {
					exl3e_half_step<KB, KA>( w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist );
				} break;
				default: {
					exl3e_half_step<KB, KB>( w, temp_costs_inc, temp_costs, hist_row, first, pre_state, write_hist );
				} break;
			}
			__syncthreads();
		}
	};

	auto backward = [&]( int roll, bool write, int edge ) {
		if ( thread == 0 ) {
			for ( int i = 255; i >= 0; --i ) {
				const int ri = exl3e_ring( i, roll );
				const int prev_edge = (int) temp_edges[(size_t) edges_max * ri + edge];
				const int kin = KA + ( ( MASK >> ( ri & 15 ) ) & 1 );
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

	forward( 128, -1 );
	const int end_state = backward( 128, false, exl3e_argmin<NT>( temp_costs, edges_last, sh_idx ) );
	forward( 0, end_state );
	backward( 0, true, end_state );
}
