// CPU check of the tensor-core fragment maps, no GPU needed: the generic m16n8k16 maps (src/kernels/cuda/mma_frag.h) and the NVFP4
// skinny / MoE ones (src/kernels/cuda/nvfp4/nvfp4_frag.h). A warp is simulated as 32 register files; mma.m8n8k4 and mma.m16n8k16 are
// modelled from the PTX ISA fragment layout tables alone, so the maps are checked against the instruction definitions, not against
// themselves. Values are small integers, so every sum is exact.
//   0. mma_frag.h: A / B fragments gathered by the maps (as lds_frag_a / lds_frag_b load them) and C scattered by them reproduce A B
//   4. mma_emu.h (the ROCm forms): m16n8k16 (FP32 and FP16 accumulators), m16n8k8 and m8n8k4 run lane by lane on the simulated warp
//      (each lane's shuffles read the other lanes' registers) and reproduce the layout models; its FP16 conversions are exact
//   1. SM_70 path (lane row / own column, 4 x m8n8k4): reproduces the reference, which validates the m8n8k4 model against the
//      mapping the V100 kernels already rely on
//   2. SM_80+ path (operands swapped, 2 x m16n8k16) through the nvfp4_afrag / bfrag / acc maps: reproduces the reference
//   3. nvfp4_pair_bits(q, p) equals dequant8_tm's element p (the same bit operations on the unshifted word)
#include "../../src/kernels/cuda/mma_frag.h"
#include "../../src/kernels/cuda/mma_emu.h"
#include "../../src/kernels/cuda/nvfp4/nvfp4_frag.h"
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>

static int failures = 0;
#define CHECK( cond, ... ) do { if ( !( cond ) ) { if ( failures < 10 ) { std::printf( "[FAIL] " __VA_ARGS__ ); std::printf( "\n" ); } ++failures; } } while ( 0 )

// One step's operands: token rows X[8][16], weight columns W[32][16] (W[c][k] = weight of output column c at k)
static float X[8][16], W[32][16], ref[8][32];

// Weight value of column c at k as the prepacked layout stores it: slot nvfp4_col_slot(c), k = 8w + 2p + h -> word w, nibble p + 4h
static float w_at_slot( int slot, int k ) { return W[nvfp4_slot_col( slot )][k]; }

// mma.m8n8k4.row.col f32: quad pair q = lanes 4q..4q+3 and 4q+16..4q+19 multiplies its own A (8x4) by its own B (4x8).
// A: lane row = (lane % 4) + 4 * (lane >= 16), a_i = column i. B: lane column = the same, b_i = row i.
// D f32 register i: row = (lane % 2) + (i & 2) + 4 * (lane >= 16), column = (i & 1) + (lane & 2) + (i & 4)
static void mma_m8n8k4( const float a[32][4], const float b[32][4], float d[32][8] ) {
	for ( int q = 0; q < 4; ++q ) {
		float A[8][4], B[4][8];
		for ( int hi = 0; hi < 2; ++hi )
			for ( int l = 0; l < 4; ++l ) {
				const int lane = 4 * q + l + 16 * hi, rc = l + 4 * hi;
				for ( int i = 0; i < 4; ++i ) { A[rc][i] = a[lane][i]; B[i][rc] = b[lane][i]; }
			}
		for ( int hi = 0; hi < 2; ++hi )
			for ( int l = 0; l < 4; ++l ) {
				const int lane = 4 * q + l + 16 * hi;
				for ( int i = 0; i < 8; ++i ) {
					const int row = ( lane % 2 ) + ( i & 2 ) + 4 * hi, col = ( i & 1 ) + ( lane & 2 ) + ( i & 4 );
					for ( int k = 0; k < 4; ++k ) d[lane][i] += A[row][k] * B[k][col];
				}
			}
	}
}

// mma.m16n8k16.row.col f32, g = lane / 4, t = lane % 4. A (16x16), element pairs: a0 (g, 2t..2t+1), a1 (g+8, 2t..), a2 (g, 2t+8..),
// a3 (g+8, 2t+8..). B (16x8): b0 (k 2t..2t+1, n g), b1 (k 2t+8.., n g). C: c0, c1 (g, 2t / 2t+1), c2, c3 (g+8, 2t / 2t+1)
static void mma_m16n8k16( const float a[32][4][2], const float b[32][2][2], float c[32][4] ) {
	float A[16][16], B[16][8];
	for ( int lane = 0; lane < 32; ++lane ) {
		const int g = lane / 4, t = lane % 4;
		for ( int e = 0; e < 2; ++e ) {
			A[g][2 * t + e] = a[lane][0][e];
			A[g + 8][2 * t + e] = a[lane][1][e];
			A[g][2 * t + 8 + e] = a[lane][2][e];
			A[g + 8][2 * t + 8 + e] = a[lane][3][e];
			B[2 * t + e][g] = b[lane][0][e];
			B[2 * t + 8 + e][g] = b[lane][1][e];
		}
	}
	for ( int lane = 0; lane < 32; ++lane ) {
		const int g = lane / 4, t = lane % 4;
		for ( int r = 0; r < 4; ++r ) {
			const int row = g + 8 * ( r >> 1 ), col = 2 * t + ( r & 1 );
			for ( int k = 0; k < 16; ++k ) c[lane][r] += A[row][k] * B[k][col];
		}
	}
}

static void check_sm70() {
	// Lane: token row r, column = its slot's; the four m8n8k4 of a k16 step take k 0-3, 4-7, 8-11, 12-15
	float d[32][8] = {};
	for ( int s = 0; s < 4; ++s ) {
		float a[32][4], b[32][4];
		for ( int lane = 0; lane < 32; ++lane ) {
			const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
			for ( int i = 0; i < 4; ++i ) { a[lane][i] = X[r][4 * s + i]; b[lane][i] = w_at_slot( lane, 4 * s + i ); }
		}
		mma_m8n8k4( a, b, d );
	}
	for ( int lane = 0; lane < 32; ++lane )
		for ( int i = 0; i < 8; ++i ) {
			const int row = nvfp4_acc_row_sm70( i, lane ), col = nvfp4_acc_col_sm70( i, lane );
			CHECK( d[lane][i] == ref[row][col], "sm70 lane %d reg %d -> (%d, %d): %g, reference %g", lane, i, row, col, d[lane][i], ref[row][col] );
		}
}

static void check_sm80() {
	float acc[32][8] = {};
	for ( int j = 0; j < 2; ++j ) {
		float a[32][4][2], b[32][2][2], c[32][4] = {};
		for ( int lane = 0; lane < 32; ++lane ) {
			for ( int r = 0; r < 4; ++r ) {
				const int slot = nvfp4_afrag_slot( j, r, lane ), p = nvfp4_afrag_kpair( r, lane );
				for ( int e = 0; e < 2; ++e ) a[lane][r][e] = w_at_slot( slot, 2 * p + e );
			}
			for ( int r = 0; r < 2; ++r ) {
				const int p = nvfp4_bfrag_kpair( r, lane );
				for ( int e = 0; e < 2; ++e ) b[lane][r][e] = X[nvfp4_bfrag_row( lane )][2 * p + e];
			}
		}
		mma_m16n8k16( a, b, c );
		for ( int lane = 0; lane < 32; ++lane )
			for ( int r = 0; r < 4; ++r ) acc[lane][4 * j + r] = c[lane][r];
	}
	int seen[8][32] = {};
	for ( int lane = 0; lane < 32; ++lane )
		for ( int i = 0; i < 8; ++i ) {
			const int row = nvfp4_acc_row_sm80( i, lane ), col = nvfp4_acc_col_sm80( i, lane );
			CHECK( row >= 0 && row < 8 && col >= 0 && col < 32, "sm80 lane %d reg %d out of range (%d, %d)", lane, i, row, col );
			if ( row < 0 || row >= 8 || col < 0 || col >= 32 ) continue;
			++seen[row][col];
			CHECK( acc[lane][i] == ref[row][col], "sm80 lane %d reg %d -> (%d, %d): %g, reference %g", lane, i, row, col, acc[lane][i], ref[row][col] );
		}
	for ( int row = 0; row < 8; ++row )
		for ( int col = 0; col < 32; ++col ) CHECK( seen[row][col] == 1, "sm80 output (%d, %d) written %d times", row, col, seen[row][col] );
}

static void check_generic( uint32_t& seed ) {
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return static_cast<float>( static_cast<int>( ( seed >> 16 ) % 15 ) - 7 ); };
	float A[16][16], Bn[8][16], R[16][8] = {};   // B stored n-major (Bn[n][k]), as the kernels keep K and V^T
	for ( int i = 0; i < 16; ++i )
		for ( int k = 0; k < 16; ++k ) A[i][k] = rnd();
	for ( int n = 0; n < 8; ++n )
		for ( int k = 0; k < 16; ++k ) Bn[n][k] = rnd();
	for ( int i = 0; i < 16; ++i )
		for ( int n = 0; n < 8; ++n )
			for ( int k = 0; k < 16; ++k ) R[i][n] += A[i][k] * Bn[n][k];
	float a[32][4][2], b[32][2][2], c[32][4] = {};
	for ( int lane = 0; lane < 32; ++lane ) {
		for ( int r = 0; r < 4; ++r )
			for ( int e = 0; e < 2; ++e ) a[lane][r][e] = A[frag_a_row( r, lane )][2 * frag_a_kpair( r, lane ) + e];
		for ( int r = 0; r < 2; ++r )
			for ( int e = 0; e < 2; ++e ) b[lane][r][e] = Bn[frag_b_col( lane )][2 * frag_b_kpair( r, lane ) + e];
	}
	mma_m16n8k16( a, b, c );
	int seen[16][8] = {};
	for ( int lane = 0; lane < 32; ++lane )
		for ( int r = 0; r < 4; ++r ) {
			const int row = frag_c_row( r, lane ), col = frag_c_col( r, lane );
			++seen[row][col];
			CHECK( c[lane][r] == R[row][col], "generic lane %d reg %d -> (%d, %d): %g, reference %g", lane, r, row, col, c[lane][r], R[row][col] );
		}
	for ( int i = 0; i < 16; ++i )
		for ( int n = 0; n < 8; ++n ) CHECK( seen[i][n] == 1, "generic output (%d, %d) written %d times", i, n, seen[i][n] );
}

// ---- mma_emu.h on the simulated warp --------------------------------------------------------------------------------------------
// Each lane's operand registers live in R[lane][0..nregs); a shuffle of register *reg from lane src reads R[src][index of reg]
// (The callable is passed by value, so it refers to the register file rather than holding it)
static uint32_t warp_regs[32][8];
struct SimWarp {
	uint32_t ( *R )[8] = warp_regs;
	uint32_t operator()( const uint32_t* reg, int src ) const {
		const ptrdiff_t off = reg - &R[0][0];
		return R[src][off % 8];
	}
};
static uint32_t pack2( float lo, float hi ) { return mma_emu_f2h( lo ) | ( mma_emu_f2h( hi ) << 16 ); }

static void check_emu( uint32_t& seed ) {
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return static_cast<float>( static_cast<int>( ( seed >> 16 ) % 15 ) - 7 ); };
	float A[16][16], B[16][8], C[16][8];
	for ( int i = 0; i < 16; ++i )
		for ( int k = 0; k < 16; ++k ) A[i][k] = rnd();
	for ( int k = 0; k < 16; ++k )
		for ( int n = 0; n < 8; ++n ) B[k][n] = rnd();
	for ( int i = 0; i < 16; ++i )
		for ( int n = 0; n < 8; ++n ) C[i][n] = rnd();
	SimWarp w;
	// m16n8k16: registers 0..3 = a, 4..5 = b
	for ( int lane = 0; lane < 32; ++lane ) {
		const int g = lane >> 2, t = lane & 3;
		for ( int r = 0; r < 4; ++r ) w.R[lane][r] = pack2( A[g + 8 * ( r & 1 )][2 * t + 8 * ( r >> 1 )], A[g + 8 * ( r & 1 )][2 * t + 8 * ( r >> 1 ) + 1] );
		for ( int r = 0; r < 2; ++r ) w.R[lane][4 + r] = pack2( B[2 * t + 8 * r][g], B[2 * t + 8 * r + 1][g] );
	}
	for ( int lane = 0; lane < 32; ++lane ) {
		const int g = lane >> 2, t = lane & 3;
		float c[4];
		for ( int r = 0; r < 4; ++r ) c[r] = C[g + 8 * ( r >> 1 )][2 * t + ( r & 1 )];
		mma_emu_m16n8k16_f32( &w.R[lane][0], &w.R[lane][4], c, lane, w );
		uint32_t ch[2] = { pack2( C[g][2 * t], C[g][2 * t + 1] ), pack2( C[g + 8][2 * t], C[g + 8][2 * t + 1] ) };
		mma_emu_m16n8k16_f16( &w.R[lane][0], &w.R[lane][4], ch, lane, w );
		for ( int r = 0; r < 4; ++r ) {
			const int row = g + 8 * ( r >> 1 ), col = 2 * t + ( r & 1 );
			float ref = C[row][col];
			for ( int k = 0; k < 16; ++k ) ref += A[row][k] * B[k][col];
			CHECK( c[r] == ref, "emu m16n8k16 f32 lane %d reg %d: %g, reference %g", lane, r, c[r], ref );
			const float h = ( r & 1 ) ? mma_emu_hi( ch[r >> 1] ) : mma_emu_lo( ch[r >> 1] );
			CHECK( h == ref, "emu m16n8k16 f16 lane %d reg %d: %g, reference %g", lane, r, h, ref );
		}
	}
	// m16n8k8 (A: registers 0..1 = rows g, g + 8 at k pair t; B: register 2 = k pair t of column g), the first eight k
	for ( int lane = 0; lane < 32; ++lane ) {
		const int g = lane >> 2, t = lane & 3;
		w.R[lane][0] = pack2( A[g][2 * t], A[g][2 * t + 1] );
		w.R[lane][1] = pack2( A[g + 8][2 * t], A[g + 8][2 * t + 1] );
		w.R[lane][2] = pack2( B[2 * t][g], B[2 * t + 1][g] );
	}
	for ( int lane = 0; lane < 32; ++lane ) {
		const int g = lane >> 2, t = lane & 3;
		uint32_t ch[2] = { pack2( C[g][2 * t], C[g][2 * t + 1] ), pack2( C[g + 8][2 * t], C[g + 8][2 * t + 1] ) };
		mma_emu_m16n8k8_f16( &w.R[lane][0], &w.R[lane][2], ch, lane, w );
		for ( int r = 0; r < 4; ++r ) {
			const int row = g + 8 * ( r >> 1 ), col = 2 * t + ( r & 1 );
			float ref = C[row][col];
			for ( int k = 0; k < 8; ++k ) ref += A[row][k] * B[k][col];
			const float h = ( r & 1 ) ? mma_emu_hi( ch[r >> 1] ) : mma_emu_lo( ch[r >> 1] );
			CHECK( h == ref, "emu m16n8k8 lane %d reg %d: %g, reference %g", lane, r, h, ref );
		}
	}
	// m8n8k4: each quad pair q multiplies its own 8 x 4 A (rows of A shifted by q) and 4 x 8 B; registers 0..1 = a, 2..3 = b
	for ( int lane = 0; lane < 32; ++lane ) {
		const int q = ( lane >> 2 ) & 3, e = ( lane & 3 ) + 4 * ( lane >> 4 );
		w.R[lane][0] = pack2( A[e + q][0 + q], A[e + q][1 + q] );
		w.R[lane][1] = pack2( A[e + q][2 + q], A[e + q][3 + q] );
		w.R[lane][2] = pack2( B[0 + q][e], B[1 + q][e] );
		w.R[lane][3] = pack2( B[2 + q][e], B[3 + q][e] );
	}
	for ( int lane = 0; lane < 32; ++lane ) {
		const int q = ( lane >> 2 ) & 3;
		float d[8];
		for ( int i = 0; i < 8; ++i ) d[i] = C[i][i & 7];
		mma_emu_m8n8k4_f32( &w.R[lane][0], &w.R[lane][2], d, lane, w );
		for ( int i = 0; i < 8; ++i ) {
			const int row = ( lane & 1 ) + ( i & 2 ) + 4 * ( lane >> 4 ), col = ( i & 1 ) + ( lane & 2 ) + ( i & 4 );
			float ref = C[i][i & 7];
			for ( int k = 0; k < 4; ++k ) ref += A[row + q][k + q] * B[k + q][col];
			CHECK( d[i] == ref, "emu m8n8k4 lane %d reg %d -> (%d, %d): %g, reference %g", lane, i, row, col, d[i], ref );
		}
	}
}

// The FP16 helpers: every finite half round-trips; float -> half rounds to nearest even (checked against the neighbouring halves)
static void check_half_conversions() {
	for ( uint32_t h = 0; h < 0x10000u; ++h ) {
		if ( ( h & 0x7c00u ) == 0x7c00u ) continue;
		const float f = mma_emu_h2f( h );
		CHECK( mma_emu_f2h( f ) == h, "half %04x -> %g -> %04x", h, f, mma_emu_f2h( f ) );
		if ( ( h & 0x7fffu ) < 0x7bffu ) {   // the midpoint to the next half up rounds to the even one of the two
			const float mid = 0.5f * ( f + mma_emu_h2f( h + 1 ) );
			const uint32_t want = ( h & 1u ) ? h + 1 : h;
			if ( ( h & 0x8000u ) == 0 ) CHECK( mma_emu_f2h( mid ) == want, "midpoint above %04x -> %04x, want %04x", h, mma_emu_f2h( mid ), want );
		}
	}
	CHECK( mma_emu_f2h( 65520.0f ) == 0x7c00u && mma_emu_f2h( 65519.0f ) == 0x7bffu, "half overflow threshold" );
}

// dequant8_tm's bit operations (src/kernels/cuda/nvfp4/nvfp4_common.cuh), before the scale multiply
static void dequant8_bits( unsigned q, unsigned out[4] ) {
	constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
	out[0] = ( ( q << 12 ) & S ) | ( ( q << 9 ) & EM );
	out[1] = ( ( q << 8 ) & S ) | ( ( q << 5 ) & EM );
	out[2] = ( ( q << 4 ) & S ) | ( ( q << 1 ) & EM );
	out[3] = ( q & S ) | ( ( q >> 3 ) & EM );
}

int main() {
	uint32_t seed = 12345;
	auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return static_cast<int>( ( seed >> 16 ) % 15 ) - 7; };
	for ( int trial = 0; trial < 20; ++trial ) {
		for ( int r = 0; r < 8; ++r )
			for ( int k = 0; k < 16; ++k ) X[r][k] = static_cast<float>( rnd() );
		for ( int c = 0; c < 32; ++c )
			for ( int k = 0; k < 16; ++k ) W[c][k] = static_cast<float>( rnd() );
		for ( int r = 0; r < 8; ++r )
			for ( int c = 0; c < 32; ++c ) {
				float s = 0.f;
				for ( int k = 0; k < 16; ++k ) s += X[r][k] * W[c][k];
				ref[r][c] = s;
			}
		check_sm70();
		check_sm80();
		check_generic( seed );
		check_emu( seed );
	}
	check_half_conversions();
	for ( int L = 0; L < 32; ++L ) CHECK( nvfp4_col_slot( nvfp4_slot_col( L ) ) == L, "slot %d does not round-trip", L );
	for ( int trial = 0; trial < 100000; ++trial ) {
		seed = seed * 1664525u + 1013904223u;
		const unsigned q = seed ^ ( seed << 7 );
		unsigned out[4];
		dequant8_bits( q, out );
		for ( int p = 0; p < 4; ++p ) CHECK( nvfp4_pair_bits( q, p ) == out[p], "pair bits q=%08x p=%d", q, p );
	}
	std::printf( "%s: %d failures\n", failures ? "FAIL" : "PASS", failures );
	return failures ? 1 : 0;
}
