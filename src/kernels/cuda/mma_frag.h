#pragma once
// mma.m16n8k16 (.row.col, f16 inputs) fragment maps: for each register, which element of the tile a lane holds (g = lane >> 2,
// t = lane & 3). Register pairs hold two consecutive k (A, B) or n (C) elements. Host + device so tests/2_parity/test_mma_frag.cpp
// checks them against the PTX ISA tables on the CPU; mma_sm80.cuh loads shared-memory fragments with them.
//   A (16 x 16, row major): register r (0..3) holds row frag_a_row, k pair frag_a_kpair (k = 2p, 2p + 1)
//   B (16 x 8, column major): register r (0..1) holds column (n) frag_b_col, k pair frag_b_kpair
//   C / D (16 x 8, fp32): register r (0..3) holds row frag_c_row, column frag_c_col

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define MMA_FRAG_HD __host__ __device__ __forceinline__
#else
	#define MMA_FRAG_HD inline
#endif

MMA_FRAG_HD int frag_a_row( int r, int lane ) { return ( lane >> 2 ) + 8 * ( r & 1 ); }
MMA_FRAG_HD int frag_a_kpair( int r, int lane ) { return ( lane & 3 ) + 4 * ( r >> 1 ); }
MMA_FRAG_HD int frag_b_col( int lane ) { return lane >> 2; }
MMA_FRAG_HD int frag_b_kpair( int r, int lane ) { return ( lane & 3 ) + 4 * r; }
MMA_FRAG_HD int frag_c_row( int r, int lane ) { return ( lane >> 2 ) + 8 * ( r >> 1 ); }
MMA_FRAG_HD int frag_c_col( int r, int lane ) { return 2 * ( lane & 3 ) + ( r & 1 ); }
