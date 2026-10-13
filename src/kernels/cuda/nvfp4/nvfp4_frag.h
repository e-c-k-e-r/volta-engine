#pragma once
// Fragment index maps of the NVFP4 skinny / MoE kernels: which lane holds what in the prepacked weight layout, and where each
// accumulator register lands. Host + device so tests/2_parity/test_nvfp4_frag.cpp checks them against the PTX fragment layouts on the CPU.
//
// Prepacked layout: per 32-column tile and k16 group, 32 lane slots of uint2 codes (+ one E4M3 scale byte). Slot L holds column
// qp * 8 + r of the tile (qp = (L >> 2) & 3, r = (L & 3) + 4 * (L >= 16): the Volta m8n8k4 B layout), its 16 codes in two words: word
// w, nibbles (p, p + 4) are k = 8w + 2p and 8w + 2p + 1.
//
// One warp step multiplies 8 token rows by the tile's 32 columns over 16 k.
//   SM_70: 4 x mma.m8n8k4, A = the token rows (lane row r), B = the lane's own column, fp32 accumulator in the m8n8k4 D layout.
//   SM_80+: 2 x mma.m16n8k16 with the operands swapped: A = 16 weight columns as rows (columns 16j..16j+15 for MMA j), B = the 8
//   tokens as columns, so the 8-row token tile fills N exactly. Lane (g = lane >> 2, t = lane & 3) needs, of MMA j, columns 16j + g
//   and 16j + g + 8 at k pairs t and t + 4, and of the tokens, token g at k pairs t and t + 4.

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define NVFP4_HD __host__ __device__ __forceinline__
#else
	#define NVFP4_HD inline
#endif

// Column of the tile held in lane slot L, and its inverse
NVFP4_HD int nvfp4_slot_col( int L ) { return ( ( L >> 2 ) & 3 ) * 8 + ( L & 3 ) + ( ( L & 16 ) ? 4 : 0 ); }
NVFP4_HD int nvfp4_col_slot( int c ) { return 4 * ( c >> 3 ) + ( c & 3 ) + ( ( c & 4 ) << 2 ); }

// SM_70 accumulator register i of lane -> (token row 0..7, column 0..31)
NVFP4_HD int nvfp4_acc_row_sm70( int i, int lane ) { return ( i & 2 ) | ( ( lane & 16 ) ? 4 : 0 ) | ( lane & 1 ); }
NVFP4_HD int nvfp4_acc_col_sm70( int i, int lane ) { return ( ( lane >> 2 ) & 3 ) * 8 + ( ( i & 1 ) | ( ( ( lane >> 1 ) & 1 ) << 1 ) | ( ( i >> 2 ) << 2 ) ); }

// SM_80+ (two m16n8k16, registers 4j..4j+3 = c0..c3 of MMA j) -> (token row, column)
NVFP4_HD int nvfp4_acc_row_sm80( int i, int lane ) { return 2 * ( lane & 3 ) + ( i & 1 ); }
NVFP4_HD int nvfp4_acc_col_sm80( int i, int lane ) { return 16 * ( i >> 2 ) + ( lane >> 2 ) + 8 * ( ( i >> 1 ) & 1 ); }

// SM_80+ A operand: lane slot to read for register a (0..3) of MMA j, and the k pair (0..7) of that slot's 16 codes it takes
NVFP4_HD int nvfp4_afrag_slot( int j, int a, int lane ) { return nvfp4_col_slot( 16 * j + ( lane >> 2 ) + 8 * ( a & 1 ) ); }
NVFP4_HD int nvfp4_afrag_kpair( int a, int lane ) { return ( lane & 3 ) + 4 * ( a >> 1 ); }
// SM_80+ B operand: token row and k pair of register b (0..1)
NVFP4_HD int nvfp4_bfrag_row( int lane ) { return lane >> 2; }
NVFP4_HD int nvfp4_bfrag_kpair( int b, int lane ) { return ( lane & 3 ) + 4 * b; }

// Two FP4 codes -> FP16 bit pairs, before the scale: nibbles (p, p + 4) of q as (low, high) half. dequant8_tm's element p equals this
// with p = 0 on q >> 4p (the masks drop the shifted-in bits)
NVFP4_HD unsigned nvfp4_pair_bits( unsigned q, int p ) {
	constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
	q >>= 4 * p;
	return ( ( q << 12 ) & S ) | ( ( q << 9 ) & EM );
}
