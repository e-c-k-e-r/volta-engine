#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include <cstdio>
#include <cstdlib>
#include "util.cuh"
#include "ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_moe_coop.cuh"
#include "trellis_context.h"

namespace exl3_moe_coop_ns {

constexpr int WK = 16;                          // warps per block
constexpr int WNT = MOE_COOP_WNT;               // adjacent n-tiles per warp
constexpr int PF = 4;                           // prefetch ring depth
constexpr int FOLD = 4;                         // fp16 -> fp32 fold cadence
constexpr int THREADS = WK * 32;
constexpr int COLS = WNT * 16;                  // columns per warp
constexpr int ROWS = 8;                         // slots of one expert per block (MMA rows)
constexpr int BITS = 3;
constexpr int TWORDS = 8 * BITS;                // uint32 per 16x16 tile
constexpr int GWORDS = WNT * TWORDS;            // uint32 per warp per k-slice
constexpr int LOADS = WNT;
constexpr int LSTRIDE = 24;
constexpr float R_SCALE = 0.088388347648f;      // 1/sqrt(128)

// Block geometry: WN warps along n (adjacent 32-column pieces), WK / WN along k
template <int WN> constexpr int tile_wn() { return WN; }
template <int WN> constexpr int tile_wk() { return WK / WN; }
template <int WN> constexpr int tile_cols() { return WN * COLS; }
template <int WN> constexpr int tile_gpc() { return tile_cols<WN>() >= 128 ? 1 : 128 / tile_cols<WN>(); }  // block groups per 128-chunk
template <int WN> constexpr int tile_cpb() { return tile_cols<WN>() >= 128 ? tile_cols<WN>() / 128 : 1; }  // 128-chunks per block

constexpr int smem_red_bytes() { return WK * ROWS * COLS * 4; }
constexpr int smem_part_bytes() { return WK * 128 * 4; }

__device__ __forceinline__ float act_silu( float x ) {
	return x * __fdividef( 1.0f, 1.0f + __expf( -x ) );
}

// Gated activation with the swiglu limit of the persistent kernel: up clamped both ways, the activated gate from above
__device__ __forceinline__ float act_gate( float g, float u, float limit ) {
	float x = act_silu( g );
	if ( limit != 0.0f ) {
		u = fminf( fmaxf( u, -limit ), limit );
		x = fminf( x, limit );
	}
	return x * u;
}

// 128-element Hadamard across the warp on 4 values per lane
__device__ __forceinline__ void had128( float& v0, float& v1, float& v2, float& v3, int lane ) {
	const float s0 = v0 + v1;
	const float d0 = v0 - v1;
	const float s1 = v2 + v3;
	const float d1 = v2 - v3;
	float h0 = s0 + s1;
	float h1 = d0 + d1;
	float h2 = s0 - s1;
	float h3 = d0 - d1;
	shuffle_had_f4x32( h0, h1, h2, h3, lane );
	v0 = h0 * R_SCALE;
	v1 = h1 * R_SCALE;
	v2 = h2 * R_SCALE;
	v3 = h3 * R_SCALE;
}

__device__ __forceinline__ void unpack_h4( uint2 u, float& v0, float& v1, float& v2, float& v3 ) {
	const half2 a = *reinterpret_cast<half2*>( &u.x );
	const half2 b = *reinterpret_cast<half2*>( &u.y );
	v0 = __low2float( a );
	v1 = __high2float( a );
	v2 = __low2float( b );
	v3 = __high2float( b );
}

__device__ __forceinline__ void load_h4( const half* p, float& v0, float& v1, float& v2, float& v3 ) {
	unpack_h4( *reinterpret_cast<const uint2*>( p ), v0, v1, v2, v3 );
}

// L2-coherent loads for data written by other blocks of the same launch
__device__ __forceinline__ void load_f4_cg( const float* p, float& v0, float& v1, float& v2, float& v3 ) {
	const float4 f = __ldcg( reinterpret_cast<const float4*>( p ) );
	v0 = f.x; v1 = f.y; v2 = f.z; v3 = f.w;
}

__device__ __forceinline__ void store_h4( half* p, float v0, float v1, float v2, float v3 ) {
	const half2 a = __floats2half2_rn( v0, v1 );
	const half2 b = __floats2half2_rn( v2, v3 );
	uint2 u;
	u.x = *reinterpret_cast<const uint32_t*>( &a );
	u.y = *reinterpret_cast<const uint32_t*>( &b );
	*reinterpret_cast<uint2*>( p ) = u;
}

__device__ __forceinline__ void scale_h4( const half* p, float& v0, float& v1, float& v2, float& v3 ) {
	float s0, s1, s2, s3;
	load_h4( p, s0, s1, s2, s3 );
	v0 *= s0; v1 *= s1; v2 *= s2; v3 *= s3;
}

struct SlotInfo {
	int row;
	int local;
	float w;
	bool active;
};

__device__ __forceinline__ SlotInfo slot_info( const MoeCoopParams& p, int s ) {
	SlotInfo si;
	si.row = s / p.topk;
	int e = p.sel[s];
	si.w = __half2float( p.rw[s] );
	const bool ok = si.w != 0.0f && e >= p.min_expert && e < p.max_expert;
	si.local = ok ? e - p.min_expert : 0;
	si.active = ok;
	return si;
}

// Slot grouping (rows > 1): active slots sorted by (local expert, slot) and cut into runs of at most ROWS slots of one expert,
// built once per call by block 0 of the rotation kernel into the run table [n_runs, -, run_start[0..], order[0..]]
__device__ __forceinline__ void build_runs( const MoeCoopParams& p, int16_t* sh_order, int16_t* sh_local, int* sh_scan, int* sh_res ) {
	const int slots = p.bsz * p.topk;
	const int t = threadIdx.x;
	const int lane = t % 32;
	const int warp = t / 32;
	int* run_start = p.runs + 2;
	int* order = p.runs + 2 + MOE_COOP_MAX_SLOTS + 1;

	if ( t < slots ) {
		const SlotInfo si = slot_info( p, t );
		sh_local[t] = si.active ? static_cast<int16_t>( si.local ) : static_cast<int16_t>( -1 );
	}
	if ( t == 0 ) sh_res[0] = 0;
	__syncthreads();

	const int my_e = t < slots ? sh_local[t] : -1;
	if ( my_e >= 0 ) {
		const int key = ( my_e << 8 ) | t;
		int rank = 0;
		for ( int u = 0; u < slots; ++u ) {
			const int eu = sh_local[u];
			rank += ( eu >= 0 && ( ( eu << 8 ) | u ) < key ) ? 1 : 0;
		}
		sh_order[rank] = static_cast<int16_t>( t );
		atomicMax( sh_res, rank + 1 );
	}
	__syncthreads();
	const int n_active = sh_res[0];

	int is_start = 0;
	if ( t < n_active ) {
		const int e_t = sh_local[sh_order[t]];
		int first = 0;
		for ( int u = 0; u < slots; ++u ) {
			const int eu = sh_local[u];
			first += ( eu >= 0 && eu < e_t ) ? 1 : 0;
		}
		is_start = ( t == 0 || sh_local[sh_order[t - 1]] != e_t || ( t - first ) % ROWS == 0 ) ? 1 : 0;
		order[t] = sh_order[t];
	}
	int v = is_start;
	#pragma unroll
	for ( int o = 1; o < 32; o <<= 1 ) {
		const int n = __shfl_up_sync( 0xffffffffu, v, o );
		if ( lane >= o ) v += n;
	}
	if ( lane == 31 ) sh_scan[warp] = v;
	__syncthreads();
	int base = 0, total_runs = 0;
	for ( int w = 0; w < WK; ++w ) {
		if ( w < warp ) base += sh_scan[w];
		total_runs += sh_scan[w];
	}
	const int run_id = base + v - is_start;
	if ( is_start ) run_start[run_id] = t;
	if ( t == 0 ) {
		run_start[total_runs] = n_active;
		p.runs[0] = total_runs;
	}
}

// The block's run: its slot indices into rows[] (shared) and the row count; false past the end
__device__ __forceinline__ bool read_run( const MoeCoopParams& p, int run_idx, int* rows, int& nrows ) {
	if ( !p.a_global ) {
		// One row: a run is a slot (inactive slots have no work)
		if ( run_idx >= p.bsz * p.topk ) return false;
		const SlotInfo si = slot_info( p, run_idx );
		if ( !si.active ) return false;
		if ( threadIdx.x == 0 ) rows[0] = run_idx;
		nrows = 1;
		__syncthreads();
		return true;
	}
	const int n_runs = p.runs[0];
	if ( run_idx >= n_runs ) return false;
	const int start = p.runs[2 + run_idx];
	const int end = p.runs[2 + run_idx + 1];
	nrows = end - start;
	if ( threadIdx.x < nrows ) rows[threadIdx.x] = p.runs[2 + MOE_COOP_MAX_SLOTS + 1 + start + threadIdx.x];
	__syncthreads();
	return true;
}

// GEMV tile: one column group of an (nrows, k) x (k, n) product, the rows being the run's slots (row r reads A at
// A2 + rows[r] * a_stride2 half2 and writes C at rows[r] * c_stride), the whole block splitting k across its warps.
// Body follows exl3_gemv_kernel (MMODE 1, CFG 0) for 3 bpw with register dequant.
template <int cb, int WN_>
__device__ __forceinline__ void gemv_tile(
	const uint32_t* __restrict__ B32,
	const half2* __restrict__ A2,
	size_t a_stride2,
	const int* __restrict__ rows,
	int nrows,
	float* __restrict__ C,
	size_t c_stride,
	int k_begin,                        // k-slice range of this block (split-k across blocks)
	int k_end,
	int ntiles,
	int group,                          // block column group of tile_cols<WN_>() columns
	float* __restrict__ sh_red          // [WKK][ROWS][TCOLS]
) {
	constexpr int WN = WN_;
	constexpr int WKK = tile_wk<WN_>();
	constexpr int TCOLS = tile_cols<WN_>();

	const int warp = threadIdx.x / 32;
	const int lane = threadIdx.x % 32;
	const int wn = warp % WN;                   // adjacent warps read adjacent column pieces
	const int wk = warp / WN;

	const int kslices = k_end - k_begin;
	const int chunk = CEIL_DIVIDE( kslices, WKK );
	const int ks0 = k_begin + wk * chunk;
	const int myn = max( 0, min( chunk, k_end - ks0 ) );
	const size_t slice_stride = static_cast<size_t>( ntiles ) * TWORDS;

	// A fragment row of this lane: rows[lane / 4] of the run
	const int r0 = lane >> 2;
	const bool r0_ok = r0 < nrows;
	const size_t a_row0 = r0_ok ? static_cast<size_t>( rows[r0] ) * a_stride2 : 0;
	const half2 hzero = __half2half2( __ushort_as_half( 0 ) );

	int x_src_a, x_src_b, x_s2;
	{
		const int t_offset = lane << 3;
		const int b1 = ( t_offset + 257 ) * 3;
		const int b2 = b1 + 21;
		const int i0 = ( b1 - 16 ) / 32;
		const int i2 = ( b2 - 1 ) / 32;
		x_s2 = ( i2 + 1 ) * 32 - b2;
		x_src_a = i0 % 24;
		x_src_b = i2 % 24;
	}

	const uint32_t* bp = B32 + static_cast<size_t>( ks0 ) * slice_stride + ( group * WN + wn ) * GWORDS + lane;

	auto ld_b = [&]( int i, int l ) -> uint32_t {
		return lane < 24 ? __ldcs( bp + static_cast<size_t>( i ) * slice_stride + l * LSTRIDE ) : 0;
	};

	uint32_t pf[PF][LOADS];
	#pragma unroll
	for ( int d = 0; d < PF; ++d ) {
		if ( d < myn ) {
			#pragma unroll
			for ( int l = 0; l < LOADS; ++l ) pf[d][l] = ld_b( d, l );
		}
	}

	FragC_h ch[WNT][2] = {};
	float2 acc0[WNT][2] = {};

	for ( int ib = 0; ib < myn; ib += PF ) {
		#pragma unroll
		for ( int d = 0; d < PF; ++d ) {
			const int i = ib + d;
			if ( i >= myn ) break;

			uint32_t bw[LOADS];
			#pragma unroll
			for ( int l = 0; l < LOADS; ++l ) bw[l] = pf[d][l];

			if ( i + PF < myn ) {
				#pragma unroll
				for ( int l = 0; l < LOADS; ++l ) pf[d][l] = ld_b( i + PF, l );
			}

			// A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
			const size_t a_col = static_cast<size_t>( ks0 + i ) * 8 + ( lane & 3 );
			FragB a01, a23;
			a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
			a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
			a01[1] = hzero;
			a23[1] = hzero;

			#pragma unroll
			for ( int t = 0; t < WNT; ++t ) {
				FragB f0, f1;
				const uint32_t awv = __shfl_sync( 0xffffffffu, bw[t], x_src_a );
				const uint32_t bwv = __shfl_sync( 0xffffffffu, bw[t], x_src_b );
				exl3_gemv_ns::dq8_regs_3bits<cb>( awv, bwv, x_s2, f0, f1 );
				exl3_gemv_ns::mma_ab_h( a01, a23, f0, ch[t][0] );
				exl3_gemv_ns::mma_ab_h( a01, a23, f1, ch[t][1] );
			}

			if ( ( d + 1 ) % FOLD == 0 || i + 1 == myn ) {
				#pragma unroll
				for ( int t = 0; t < WNT; ++t ) {
					#pragma unroll
					for ( int f = 0; f < 2; ++f ) {
						acc0[t][f].x += __low2float( ch[t][f][0] );
						acc0[t][f].y += __high2float( ch[t][f][0] );
						ch[t][f][0] = hzero;
					}
				}
			}
		}
	}

	// Cross-warp reduction over the k splits. Lane l holds row l/4, cols tile*16 + frag*8 + 2*(l%4) (+1)
	if ( r0_ok ) {
		const int c0 = 2 * ( lane & 3 );
		float* red = sh_red + ( wk * ROWS + r0 ) * TCOLS + wn * COLS;
		#pragma unroll
		for ( int t = 0; t < WNT; ++t ) {
			#pragma unroll
			for ( int f = 0; f < 2; ++f ) {
				const int col = t * 16 + f * 8 + c0;
				red[col + 0] = acc0[t][f].x;
				red[col + 1] = acc0[t][f].y;
			}
		}
	}
	__syncthreads();

	for ( int o = threadIdx.x; o < TCOLS * nrows; o += THREADS ) {
		const int r = o / TCOLS;
		const int c = o % TCOLS;
		float sum = 0.0f;
		#pragma unroll
		for ( int j = 0; j < WKK; ++j ) sum += sh_red[( j * ROWS + r ) * TCOLS + c];
		C[static_cast<size_t>( rows[r] ) * c_stride + group * TCOLS + c] = sum;
	}
	__syncthreads();
}

// Completion counter: every thread fences its own stores, then one thread arrives. Returns true (block-uniform) for the
// last arriving block, which may then read the other blocks' results through L2 (load_*_cg)
__device__ __forceinline__ bool arrive_last( int* counter, int expected, int* sh_flag ) {
	__threadfence();
	__syncthreads();
	if ( threadIdx.x == 0 ) *sh_flag = ( atomicAdd( counter, 1 ) == expected - 1 );
	__syncthreads();
	const bool last = *sh_flag;
	if ( last ) __threadfence();
	return last;
}

// Input rotation of one 128-chunk of a token's x row: * suh, Hadamard
__device__ __forceinline__ void rotate_chunk( const MoeCoopParams& p, int row, const half* suh, int c, half* dst, int lane ) {
	const int col = c * 128 + lane * 4;
	float v0, v1, v2, v3;
	load_h4( p.x + static_cast<size_t>( row ) * p.H + col, v0, v1, v2, v3 );
	scale_h4( suh + col, v0, v1, v2, v3 );
	had128( v0, v1, v2, v3, lane );
	store_h4( dst + col, v0, v1, v2, v3 );
}

// Rotation pre-kernel (rows > 1): warp per (slot, chunk, projection) into had_g / had_u; block 0 also builds the run table
__global__ __launch_bounds__( THREADS )
void rot_kernel( const MoeCoopParams p ) {
	__shared__ int16_t sh_order[MOE_COOP_MAX_SLOTS];
	__shared__ int16_t sh_local[MOE_COOP_MAX_SLOTS];
	__shared__ int sh_scan[WK];
	__shared__ int sh_res[4];
	const int warp = threadIdx.x / 32;
	const int lane = threadIdx.x % 32;
	const int chunks = p.H / 128;
	const int slots = p.bsz * p.topk;
	if ( blockIdx.x == 0 ) build_runs( p, sh_order, sh_local, sh_scan, sh_res );
	const int item = blockIdx.x * ( THREADS / 32 ) + warp;
	if ( item >= slots * chunks * 2 ) return;
	const int s = item / ( chunks * 2 );
	const int rem = item % ( chunks * 2 );
	const int c = rem / 2;
	const bool is_gate = ( rem % 2 ) == 0;
	const SlotInfo si = slot_info( p, s );
	if ( !si.active ) return;
	const half* suh = ( is_gate ? p.g_suh : p.u_suh )[si.local];
	half* dst = ( is_gate ? p.had_g : p.had_u ) + static_cast<size_t>( s ) * p.H;
	rotate_chunk( p, si.row, suh, c, dst, lane );
}

// Kernel A: gate/up GEMV per (expert run, group, projection) with the chunk epilogue (output rotation, svh, activation,
// down-input rotation) by the last arrival per (slot, chunk)
template <int cb, int WN_>
__global__ __launch_bounds__( THREADS )
void a_kernel( const MoeCoopParams p ) {
	constexpr int TCOLS = tile_cols<WN_>();
	constexpr int GPC = tile_gpc<WN_>();
	constexpr int CPB = tile_cpb<WN_>();
	extern __shared__ uint32_t smem_dyn[];
	half* sh_A = reinterpret_cast<half*>( smem_dyn );
	float* sh_red = reinterpret_cast<float*>( smem_dyn + p.H / 2 );
	__shared__ int sh_flag;
	__shared__ int sh_rows[ROWS];

	const int warp = threadIdx.x / 32;
	const int lane = threadIdx.x % 32;
	const int ng = p.I / TCOLS;

	// Reset the down stage's counters for this call (its previous use finished in stream order)
	for ( int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * THREADS ) p.ctr_b[i] = 0;

	const int item = blockIdx.x;
	const bool is_gate = ( item % 2 ) == 0;
	const int rem = item / 2;
	const int ks = rem % p.ksplit_a;                 // split-k index: partial output rows
	const int rem2 = rem / p.ksplit_a;
	const int run_idx = rem2 / ng;
	const int group = rem2 % ng;
	int nrows = 0;
	if ( !read_run( p, run_idx, sh_rows, nrows ) ) return;
	const int local = slot_info( p, sh_rows[0] ).local;
	if ( p.heat != nullptr && threadIdx.x == 0 && is_gate && ks == 0 && group == 0 ) atomicAdd( &p.heat[local], static_cast<uint32_t>( nrows ) );
	const int kslices_all = p.H / 16;
	const int k_begin = kslices_all * ks / p.ksplit_a;
	const int k_end = kslices_all * ( ks + 1 ) / p.ksplit_a;

	const half* suh = ( is_gate ? p.g_suh : p.u_suh )[local];
	const half2* A2;
	size_t a_stride2;
	if ( p.a_global ) {
		A2 = reinterpret_cast<const half2*>( is_gate ? p.had_g : p.had_u );
		a_stride2 = static_cast<size_t>( p.H ) / 2;
	}
	else {
		// One row: every run is a single slot; rotate its input into shared memory here. A zero row stride makes the tile read
		// it whatever the slot index (which still selects the C row)
		const SlotInfo si = slot_info( p, sh_rows[0] );
		for ( int c = warp; c < p.H / 128; c += WK ) rotate_chunk( p, si.row, suh, c, sh_A, lane );
		__syncthreads();
		A2 = reinterpret_cast<const half2*>( sh_A );
		a_stride2 = 0;
	}

	{
		const uint32_t* B32 = reinterpret_cast<const uint32_t*>( ( is_gate ? p.g_trellis : p.u_trellis )[local] );
		// Partial output rows of split ks live at slot + ks * slots
		float* C = ( is_gate ? p.gu_g : p.gu_u ) + static_cast<size_t>( ks ) * ( p.bsz * p.topk ) * p.I;
		gemv_tile<cb, WN_>( B32, A2, a_stride2, sh_rows, nrows, C, p.I, k_begin, k_end, p.I / 16, group, sh_red );
	}

	for ( int r = 0; r < nrows; ++r ) {
		for ( int j = 0; j < CPB; ++j ) {
			const int s = sh_rows[r];
			const int chunk = ( group * CPB + j ) / GPC;
			if ( !arrive_last( p.ctr_a + static_cast<size_t>( s ) * ( p.I / 128 ) + chunk, GPC * 2 * p.ksplit_a, &sh_flag ) ) continue;
			if ( warp == 0 ) {
				const int col = chunk * 128 + lane * 4;
				const size_t off = static_cast<size_t>( s ) * p.I + col;
				const size_t pstride = static_cast<size_t>( p.bsz * p.topk ) * p.I;

				// Sum of the split-k partials (fixed order)
				auto load_sum = [&]( const float* buf, float& v0, float& v1, float& v2, float& v3 ) {
					v0 = v1 = v2 = v3 = 0.0f;
					for ( int q = 0; q < p.ksplit_a; ++q ) {
						float t0, t1, t2, t3;
						load_f4_cg( buf + q * pstride + off, t0, t1, t2, t3 );
						v0 += t0; v1 += t1; v2 += t2; v3 += t3;
					}
				};
				float u0, u1, u2, u3;
				load_sum( p.gu_u, u0, u1, u2, u3 );
				had128( u0, u1, u2, u3, lane );
				scale_h4( p.u_svh[local] + col, u0, u1, u2, u3 );

				float g0, g1, g2, g3;
				load_sum( p.gu_g, g0, g1, g2, g3 );
				had128( g0, g1, g2, g3, lane );
				scale_h4( p.g_svh[local] + col, g0, g1, g2, g3 );

				float a0 = act_gate( g0, u0, p.act_limit );
				float a1 = act_gate( g1, u1, p.act_limit );
				float a2 = act_gate( g2, u2, p.act_limit );
				float a3 = act_gate( g3, u3, p.act_limit );

				scale_h4( p.d_suh[local] + col, a0, a1, a2, a3 );
				had128( a0, a1, a2, a3, lane );
				store_h4( p.act_out + off, a0, a1, a2, a3 );
			}
			__syncthreads();
		}
	}
}

// Kernel B: down GEMV per (expert run, group); the last arrival of a (token, chunk) reduces the token's slots (rotation, svh,
// routing weight) in a fixed order and stores the row
template <int cb, int WN_>
__global__ __launch_bounds__( THREADS )
void b_kernel( const MoeCoopParams p ) {
	constexpr int TCOLS = tile_cols<WN_>();
	constexpr int GPC = tile_gpc<WN_>();
	constexpr int CPB = tile_cpb<WN_>();
	extern __shared__ uint32_t smem_dyn[];
	float* sh_red = reinterpret_cast<float*>( smem_dyn );
	float* sh_part = reinterpret_cast<float*>( smem_dyn + WK * ROWS * COLS );
	__shared__ int sh_flag;
	__shared__ int sh_n_active;
	__shared__ int sh_rows[ROWS];

	const int warp = threadIdx.x / 32;
	const int lane = threadIdx.x % 32;
	const int ng = p.H / TCOLS;

	// Reset the gate/up stage's counters for the next call
	for ( int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * THREADS ) p.ctr_a[i] = 0;

	const int item = blockIdx.x;
	const int run_idx = item / ng;
	const int group = item % ng;
	int nrows = 0;
	if ( !read_run( p, run_idx, sh_rows, nrows ) ) return;
	const int local = slot_info( p, sh_rows[0] ).local;

	{
		const uint32_t* B32 = reinterpret_cast<const uint32_t*>( p.d_trellis[local] );
		gemv_tile<cb, WN_>( B32, reinterpret_cast<const half2*>( p.act_out ), static_cast<size_t>( p.I ) / 2, sh_rows, nrows, p.d_out, p.H,
			0, p.I / 16, p.H / 16, group, sh_red );
	}

	for ( int r = 0; r < nrows; ++r ) {
		for ( int j = 0; j < CPB; ++j ) {
			const int s = sh_rows[r];
			const int row = s / p.topk;
			const int chunk = ( group * CPB + j ) / GPC;

			// Active slots of this token: the number of arrivals to expect for (row, chunk)
			if ( threadIdx.x == 0 ) {
				int n = 0;
				for ( int k = 0; k < p.topk; ++k ) n += slot_info( p, row * p.topk + k ).active ? 1 : 0;
				sh_n_active = n;
			}
			__syncthreads();
			if ( !arrive_last( p.ctr_b + static_cast<size_t>( row ) * ( p.H / 128 ) + chunk, sh_n_active * GPC, &sh_flag ) ) continue;

			const int col = chunk * 128 + lane * 4;

			// Each warp takes slots warp, warp + WK, ...; per-warp partials are summed in warp order
			float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
			for ( int k = warp; k < p.topk; k += WK ) {
				const int sk = row * p.topk + k;
				const SlotInfo sj = slot_info( p, sk );
				if ( !sj.active ) continue;
				float v0, v1, v2, v3;
				load_f4_cg( p.d_out + static_cast<size_t>( sk ) * p.H + col, v0, v1, v2, v3 );
				had128( v0, v1, v2, v3, lane );
				scale_h4( p.d_svh[sj.local] + col, v0, v1, v2, v3 );
				o0 += sj.w * v0;
				o1 += sj.w * v1;
				o2 += sj.w * v2;
				o3 += sj.w * v3;
			}
			float* part = sh_part + warp * 128 + lane * 4;
			part[0] = o0; part[1] = o1; part[2] = o2; part[3] = o3;
			__syncthreads();
			if ( warp == 0 ) {
				o0 = 0.0f; o1 = 0.0f; o2 = 0.0f; o3 = 0.0f;
				#pragma unroll
				for ( int w = 0; w < WK; ++w ) {
					const float* q = sh_part + w * 128 + lane * 4;
					o0 += q[0]; o1 += q[1]; o2 += q[2]; o3 += q[3];
				}
				store_h4( p.out + static_cast<size_t>( row ) * p.H + col, o0, o1, o2, o3 );
			}
			__syncthreads();
		}
	}
}

} // namespace exl3_moe_coop_ns

namespace {

int pick_ksplit( int base, int kslices, int max_split, int num_sms ) {
	static const int forced = std::getenv( "EXL3_MOE_COOP_KSPLIT" ) ? std::atoi( std::getenv( "EXL3_MOE_COOP_KSPLIT" ) ) : 0;
	if ( forced > 0 ) return MIN( forced, max_split );
	const int want = CEIL_DIVIDE( num_sms, MAX( base, 1 ) );
	const int kcap = MAX( kslices / 64, 1 );
	return MAX( 1, MIN( MIN( want, kcap ), max_split ) );
}

} // namespace

void exl3_moe_coop_launch( const MoeCoopParams& p_in, int device, cudaStream_t stream ) {
	using namespace exl3_moe_coop_ns;
	MoeCoopParams p = p_in;
	const int slots = p.bsz * p.topk;
	ENGINE_ASSERT( p.topk == MOE_COOP_TOPK && p.bsz >= 1 && p.bsz <= MOE_COOP_MAX_ROWS, "exl3_moe_coop: row count" );
	ENGINE_ASSERT( p.H % 128 == 0 && p.I % 128 == 0, "exl3_moe_coop: widths" );
	p.a_global = p.bsz > 1;
	p.ctr_a_len = MOE_COOP_MAX_SLOTS * ( p.I / 128 );
	p.ctr_b_len = MOE_COOP_MAX_ROWS * ( p.H / 128 );

	// Block geometry per stage: warps along n (the rest split k). Gate/up has a long k and takes few columns per block; the
	// down projection's k is short, so wide column tiles keep the per-block fixed costs down
	static const int wn_a_env = std::getenv( "EXL3_MOE_COOP_WNA" ) ? std::atoi( std::getenv( "EXL3_MOE_COOP_WNA" ) ) : 4;
	static const int wn_b_env = std::getenv( "EXL3_MOE_COOP_WNB" ) ? std::atoi( std::getenv( "EXL3_MOE_COOP_WNB" ) ) : 4;
	const int wn_a = wn_a_env, wn_b = wn_b_env;
	const int num_sms = engine::modules::trellis_get_num_sms( device );
	const int tcols_a = wn_a * COLS;
	const int tcols_b = wn_b * COLS;
	ENGINE_ASSERT( p.I % tcols_a == 0 && p.H % tcols_b == 0, "exl3_moe_coop: tile width" );

	// The split-k factor is sized for the one-row grid and kept for every row count: it sets the order in which a row's partial
	// sums are added, and a row's result must not depend on how many other rows share the call (speculative verify == plain decode)
	const int base_a = slots * 2 * ( p.I / tcols_a );
	p.ksplit_a = pick_ksplit( MOE_COOP_TOPK * 2 * ( p.I / tcols_a ), p.H / 16, MOE_COOP_KSPLIT_MAX, num_sms );
	p.ksplit_b = 1;
	const int grid_a = base_a * p.ksplit_a;
	const int grid_b = slots * ( p.H / tcols_b );

	const int smem_a = p.H * 2 + smem_red_bytes();
	const int smem_b = smem_red_bytes() + smem_part_bytes();
	static bool attr_set[ENGINE_NUM_GPUS] = { false };
	if ( device >= 0 && device < ENGINE_NUM_GPUS && !attr_set[device] ) {
		CUDA_CHECK( cudaFuncSetAttribute( (void*) a_kernel<2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) a_kernel<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) a_kernel<2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) a_kernel<2, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) b_kernel<2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) b_kernel<2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) b_kernel<2, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 ) );
		CUDA_CHECK( cudaFuncSetAttribute( (void*) b_kernel<2, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024 ) );
		attr_set[device] = true;
	}

	if ( p.a_global ) {
		const int rot_warps = slots * ( p.H / 128 ) * 2;
		rot_kernel<<<dim3( MAX( 1, CEIL_DIVIDE( rot_warps, THREADS / 32 ) ) ), dim3( THREADS ), 0, stream>>>( p );
	}
	switch ( wn_a ) {
		case 1: a_kernel<2, 1><<<dim3( grid_a ), dim3( THREADS ), smem_a, stream>>>( p ); break;
		case 2: a_kernel<2, 2><<<dim3( grid_a ), dim3( THREADS ), smem_a, stream>>>( p ); break;
		case 8: a_kernel<2, 8><<<dim3( grid_a ), dim3( THREADS ), smem_a, stream>>>( p ); break;
		default: a_kernel<2, 4><<<dim3( grid_a ), dim3( THREADS ), smem_a, stream>>>( p ); break;
	}
	switch ( wn_b ) {
		case 2: b_kernel<2, 2><<<dim3( grid_b ), dim3( THREADS ), smem_b, stream>>>( p ); break;
		case 8: b_kernel<2, 8><<<dim3( grid_b ), dim3( THREADS ), smem_b, stream>>>( p ); break;
		case 16: b_kernel<2, 16><<<dim3( grid_b ), dim3( THREADS ), smem_b, stream>>>( p ); break;
		default: b_kernel<2, 4><<<dim3( grid_b ), dim3( THREADS ), smem_b, stream>>>( p ); break;
	}
	CUDA_CHECK( cudaGetLastError() );
}
