// ROCm routed-expert MoE for trellis (EXL3) experts, any bit width 1..8, mul1 codebook: the fused decode path of MoE::forward
// (exl3_moe_coop.cuh's interface and scratch) as five plain launches, without the cooperative kernels' mma.sync (emulated on ROCm) and
// without their 3 / 4 bpw instantiations (Qwen3.8-Flash-Next-exl3 stores every expert at 2 bpw). Per call, for every active slot
// (a route whose expert is local and whose routing weight is not zero), with H(.) the 128-point Hadamard scaled by 1/sqrt(128):
//   rot : had_g/u[s] = H( x[row] * suh_g/u[e] )                        (one warp per slot, 128-chunk and projection)
//   A   : gu_g/u[s]  = had_g/u[s] @ W_g/u[e]                           (the v_dot2 EXL3 GEMV of exl3_gemv.cu, one slot per block)
//   mid : act[s]     = H( act_gate( H(gu_g) * svh_g, H(gu_u) * svh_u ) * suh_d[e] )
//   B   : d_out[s]   = act[s] @ W_d[e]
//   sum : out[row]   = sum_k rw[s] * ( H(d_out[s]) * svh_d[e] )         (slots in route order, so a row does not depend on its neighbours)
// Weights are read once per slot (routes of different rows that pick one expert read it again; the infinity cache serves most of those).
#if defined(USE_ROCM)
#include <cuda_fp16.h>
#include "engine/config.h"
#include "engine/runtime/options.h"
#include "exl3_dq.cuh"
#include "exl3_moe_coop.cuh"
#include "hadamard_inner.cuh"
#include "../wmma.cuh"
#include <cstdint>
#include <cstdlib>

namespace {

constexpr float R_SCALE = 0.088388347648f;   // 1/sqrt(128)
constexpr int GEMV_WARPS = 8;

__device__ __forceinline__ float fdot2( half2 a, half2 b, float c ) {
	typedef _Float16 h2v __attribute__( ( ext_vector_type( 2 ) ) );
	return __builtin_amdgcn_fdot2( __builtin_bit_cast( h2v, a ), __builtin_bit_cast( h2v, b ), c, false );
}

// 128-element Hadamard across the warp, 4 consecutive values per lane (hadamard_inner.cuh's order), scaled
__device__ __forceinline__ void had128( float& v0, float& v1, float& v2, float& v3, int lane ) {
	const float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
	float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
	shuffle_had_f4x32( h0, h1, h2, h3, lane );
	v0 = h0 * R_SCALE; v1 = h1 * R_SCALE; v2 = h2 * R_SCALE; v3 = h3 * R_SCALE;
}

__device__ __forceinline__ void unpack4( uint2 u, float& v0, float& v1, float& v2, float& v3 ) {
	const half2 a = __builtin_bit_cast( half2, u.x ), b = __builtin_bit_cast( half2, u.y );
	v0 = __low2float( a ); v1 = __high2float( a ); v2 = __low2float( b ); v3 = __high2float( b );
}

__device__ __forceinline__ void load4( const half* p, float& v0, float& v1, float& v2, float& v3 ) {
	unpack4( *reinterpret_cast<const uint2*>( p ), v0, v1, v2, v3 );
}

__device__ __forceinline__ void store4( half* p, float v0, float v1, float v2, float v3 ) {
	const half2 a = __floats2half2_rn( v0, v1 ), b = __floats2half2_rn( v2, v3 );
	*reinterpret_cast<uint2*>( p ) = make_uint2( __builtin_bit_cast( uint32_t, a ), __builtin_bit_cast( uint32_t, b ) );
}

__device__ __forceinline__ void scale4( const half* p, float& v0, float& v1, float& v2, float& v3 ) {
	float s0, s1, s2, s3;
	load4( p, s0, s1, s2, s3 );
	v0 *= s0; v1 *= s1; v2 *= s2; v3 *= s3;
}

__device__ __forceinline__ float act_silu( float x ) { return x * __fdividef( 1.0f, 1.0f + __expf( -x ) ); }

// Gated activation with the swiglu limit: up clamped both ways, the activated gate from above
__device__ __forceinline__ float act_gate( float g, float u, float limit ) {
	float x = act_silu( g );
	if ( limit != 0.0f ) {
		u = fminf( fmaxf( u, -limit ), limit );
		x = fminf( x, limit );
	}
	return x * u;
}

// Local expert of slot s, or -1 when the route is inactive (zero weight, or an expert of another rank)
__device__ __forceinline__ int slot_local( const MoeCoopParams& p, int s ) {
	const int e = p.sel[s];
	const bool ok = __half2float( p.rw[s] ) != 0.0f && e >= p.min_expert && e < p.max_expert;
	return ok ? e - p.min_expert : -1;
}

// grid (slots, H / 128, 2), 32 threads
__global__ __launch_bounds__( 32 ) void moe_rot_kernel( const MoeCoopParams p, int* __restrict__ exp_of ) {
	const int s = blockIdx.x, chunk = blockIdx.y, proj = blockIdx.z, lane = threadIdx.x;
	const int local = slot_local( p, s );
	if ( chunk == 0 && proj == 0 && lane == 0 ) {
		exp_of[s] = local;
		if ( local >= 0 && p.heat != nullptr ) atomicAdd( &p.heat[local], 1u );
	}
	if ( local < 0 ) return;
	const int row = s / p.topk;
	const int col = chunk * 128 + lane * 4;
	const half* suh = ( proj ? p.u_suh : p.g_suh )[local];
	const uint2 xr = *reinterpret_cast<const uint2*>( p.x + static_cast<size_t>( row ) * p.H + col );
	const uint2 sr = *reinterpret_cast<const uint2*>( suh + col );
	const half2 a = __hmul2( __builtin_bit_cast( half2, xr.x ), __builtin_bit_cast( half2, sr.x ) );
	const half2 b = __hmul2( __builtin_bit_cast( half2, xr.y ), __builtin_bit_cast( half2, sr.y ) );
	float v0 = __low2float( a ), v1 = __high2float( a ), v2 = __low2float( b ), v3 = __high2float( b );
	had128( v0, v1, v2, v3, lane );
	store4( ( proj ? p.had_u : p.had_g ) + static_cast<size_t>( s ) * p.H + col, v0, v1, v2, v3 );
}

// One 16-column tile of one slot's GEMV (the EXL3 GEMV of exl3_gemv.cu for a single row): grid (N / 16, slots, projections),
// the block's warps split the k tiles. in0 / in1 are the rotated inputs of projection 0 / 1 ([slots, K]); out0 / out1 [slots, N] floats
template <int bits>
__global__ __launch_bounds__( 32 * GEMV_WARPS ) void moe_gemv_kernel(
	const uint16_t* const* __restrict__ trellis0,
	const uint16_t* const* __restrict__ trellis1,
	const half* __restrict__ in0,
	const half* __restrict__ in1,
	float* __restrict__ out0,
	float* __restrict__ out1,
	const int* __restrict__ exp_of,
	int K,
	int N
) {
	__shared__ float red[GEMV_WARPS][16];
	const int s = blockIdx.y;
	const int local = exp_of[s];
	if ( local < 0 ) return;
	const bool second = blockIdx.z != 0;
	const uint16_t* B = ( second ? trellis1 : trellis0 )[local];
	const half* A = ( second ? in1 : in0 ) + static_cast<size_t>( s ) * K;
	float* C = ( second ? out1 : out0 ) + static_cast<size_t>( s ) * N;
	const int lane = threadIdx.x & 31;
	const int warp = __builtin_amdgcn_readfirstlane( threadIdx.x >> 5 );
	const int tile_n = blockIdx.x, n_tiles = N >> 4, k_tiles = K >> 4;
	const int per = ( k_tiles + GEMV_WARPS - 1 ) / GEMV_WARPS;
	const int kb0 = warp * per, kb1 = min( k_tiles, kb0 + per );
	const half2* ar = reinterpret_cast<const half2*>( A ) + ( ( lane & 3 ) );
	float acc_a = 0.f, acc_b = 0.f;
	const uint32_t* bt = reinterpret_cast<const uint32_t*>( B + ( static_cast<size_t>( kb0 ) * n_tiles + tile_n ) * ( 16 * bits ) );
	const size_t bstep = static_cast<size_t>( n_tiles ) * 16 * bits / 2;
	#pragma unroll 4
	for ( int kt = kb0; kt < kb1; ++kt, bt += bstep ) {
		FragB f0, f1;
		dq_dispatch<bits, 2>( bt, lane << 3, f0, f1 );
		const half2 a01 = ar[kt * 8], a89 = ar[kt * 8 + 4];
		acc_a = fdot2( a01, f0[0], acc_a );
		acc_a = fdot2( a89, f0[1], acc_a );
		acc_b = fdot2( a01, f1[0], acc_b );
		acc_b = fdot2( a89, f1[1], acc_b );
	}
	acc_a += __shfl_xor( acc_a, 1 );
	acc_a += __shfl_xor( acc_a, 2 );
	acc_b += __shfl_xor( acc_b, 1 );
	acc_b += __shfl_xor( acc_b, 2 );
	if ( ( lane & 3 ) == 0 ) {
		red[warp][lane >> 2] = acc_a;
		red[warp][( lane >> 2 ) + 8] = acc_b;
	}
	__syncthreads();
	if ( threadIdx.x < 16 ) {
		float v = 0.f;
		#pragma unroll
		for ( int w = 0; w < GEMV_WARPS; ++w ) v += red[w][threadIdx.x];
		C[tile_n * 16 + threadIdx.x] = v;
	}
}

// grid (slots, I / 128), 32 threads
__global__ __launch_bounds__( 32 ) void moe_mid_kernel( const MoeCoopParams p, const int* __restrict__ exp_of ) {
	const int s = blockIdx.x, chunk = blockIdx.y, lane = threadIdx.x;
	const int local = exp_of[s];
	if ( local < 0 ) return;
	const int col = chunk * 128 + lane * 4;
	const size_t off = static_cast<size_t>( s ) * p.I + col;
	const float4 uf = *reinterpret_cast<const float4*>( p.gu_u + off );
	const float4 gf = *reinterpret_cast<const float4*>( p.gu_g + off );
	float u0 = uf.x, u1 = uf.y, u2 = uf.z, u3 = uf.w;
	float g0 = gf.x, g1 = gf.y, g2 = gf.z, g3 = gf.w;
	had128( u0, u1, u2, u3, lane );
	scale4( p.u_svh[local] + col, u0, u1, u2, u3 );
	had128( g0, g1, g2, g3, lane );
	scale4( p.g_svh[local] + col, g0, g1, g2, g3 );
	float a0 = act_gate( g0, u0, p.act_limit ), a1 = act_gate( g1, u1, p.act_limit ), a2 = act_gate( g2, u2, p.act_limit ), a3 = act_gate( g3, u3, p.act_limit );
	scale4( p.d_suh[local] + col, a0, a1, a2, a3 );
	had128( a0, a1, a2, a3, lane );
	store4( p.act_out + off, a0, a1, a2, a3 );
}

// grid (rows, H / 128), 32 threads
__global__ __launch_bounds__( 32 ) void moe_sum_kernel( const MoeCoopParams p, const int* __restrict__ exp_of ) {
	const int row = blockIdx.x, chunk = blockIdx.y, lane = threadIdx.x;
	const int col = chunk * 128 + lane * 4;
	float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
	for ( int k = 0; k < p.topk; ++k ) {
		const int s = row * p.topk + k;
		const int local = exp_of[s];
		if ( local < 0 ) continue;
		const float w = __half2float( p.rw[s] );
		const float4 d = *reinterpret_cast<const float4*>( p.d_out + static_cast<size_t>( s ) * p.H + col );
		float v0 = d.x, v1 = d.y, v2 = d.z, v3 = d.w;
		had128( v0, v1, v2, v3, lane );
		scale4( p.d_svh[local] + col, v0, v1, v2, v3 );
		o0 += w * v0; o1 += w * v1; o2 += w * v2; o3 += w * v3;
	}
	store4( p.out + static_cast<size_t>( row ) * p.H + col, o0, o1, o2, o3 );
}

template <int bits>
void launch_gemvs( const MoeCoopParams& p, int slots, const int* exp_of, cudaStream_t stream ) {
	moe_gemv_kernel<bits><<<dim3( p.I / 16, slots, 2 ), 32 * GEMV_WARPS, 0, stream>>>(
		p.g_trellis, p.u_trellis, p.had_g, p.had_u, p.gu_g, p.gu_u, exp_of, p.H, p.I );
}

template <int bits>
void launch_down( const MoeCoopParams& p, int slots, const int* exp_of, cudaStream_t stream ) {
	moe_gemv_kernel<bits><<<dim3( p.H / 16, slots, 1 ), 32 * GEMV_WARPS, 0, stream>>>(
		p.d_trellis, p.d_trellis, p.act_out, p.act_out, p.d_out, p.d_out, exp_of, p.I, p.H );
}

} // namespace

void exl3_moe_rocm_launch( const MoeCoopParams& p, cudaStream_t stream ) {
	const int slots = p.bsz * p.topk;
	ENGINE_ASSERT( slots >= 1 && slots <= MOE_COOP_MAX_SLOTS && p.H % 128 == 0 && p.I % 128 == 0 && p.bits >= 1 && p.bits <= 8, "exl3_moe_rocm: shape" );
	int* exp_of = p.runs;   // the cooperative kernel's run table, unused here: one local expert (or -1) per slot
	moe_rot_kernel<<<dim3( slots, p.H / 128, 2 ), 32, 0, stream>>>( p, exp_of );
	switch ( p.bits ) {
		case 1: launch_gemvs<1>( p, slots, exp_of, stream ); break;
		case 2: launch_gemvs<2>( p, slots, exp_of, stream ); break;
		case 3: launch_gemvs<3>( p, slots, exp_of, stream ); break;
		case 4: launch_gemvs<4>( p, slots, exp_of, stream ); break;
		case 5: launch_gemvs<5>( p, slots, exp_of, stream ); break;
		case 6: launch_gemvs<6>( p, slots, exp_of, stream ); break;
		case 7: launch_gemvs<7>( p, slots, exp_of, stream ); break;
		default: launch_gemvs<8>( p, slots, exp_of, stream ); break;
	}
	moe_mid_kernel<<<dim3( slots, p.I / 128 ), 32, 0, stream>>>( p, exp_of );
	switch ( p.bits ) {
		case 1: launch_down<1>( p, slots, exp_of, stream ); break;
		case 2: launch_down<2>( p, slots, exp_of, stream ); break;
		case 3: launch_down<3>( p, slots, exp_of, stream ); break;
		case 4: launch_down<4>( p, slots, exp_of, stream ); break;
		case 5: launch_down<5>( p, slots, exp_of, stream ); break;
		case 6: launch_down<6>( p, slots, exp_of, stream ); break;
		case 7: launch_down<7>( p, slots, exp_of, stream ); break;
		default: launch_down<8>( p, slots, exp_of, stream ); break;
	}
	moe_sum_kernel<<<dim3( p.bsz, p.H / 128 ), 32, 0, stream>>>( p, exp_of );
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Expert-grouped path for many rows (prefill chunks). The routes are sorted by local expert (deterministically: ascending slot within an
// expert), then handled in waves of W sorted positions; inside a wave a "run" is up to MR consecutive positions of one expert, and a GEMV block
// decodes the expert's weight tile once for all of the run's rows, so a cold expert is read over PCIe once per wave and not once per route.
// A row's routed output is accumulated in ascending expert order (waves go in expert order), whatever the other rows are.
//   group : order[pos] = slot, inv[slot] = pos, ex_start[e] (one block)
//   per wave: runs, rot (gate / up), gemv (gate / up), mid, gemv (down), acc
//   fin   : out[row] = half(acc[row])
namespace {

constexpr int GRP_MAX_SLOTS = 8192;
constexpr int GRP_THREADS = 1024;

__global__ __launch_bounds__( GRP_THREADS ) void grp_group_kernel( const MoeCoopParams p, int E, int* __restrict__ order, int* __restrict__ inv, int* __restrict__ ex_start ) {
	__shared__ int16_t loc[GRP_MAX_SLOTS];
	__shared__ int scan[GRP_THREADS];
	const int n = p.bsz * p.topk;
	const int tid = threadIdx.x;
	for ( int s = tid; s < n; s += GRP_THREADS ) {
		loc[s] = static_cast<int16_t>( slot_local( p, s ) );
		inv[s] = -1;
	}
	__syncthreads();
	int cnt = 0;
	if ( tid < E ) {
		for ( int s = 0; s < n; ++s ) cnt += ( loc[s] == tid ) ? 1 : 0;
	}
	scan[tid] = cnt;
	__syncthreads();
	for ( int d = 1; d < GRP_THREADS; d <<= 1 ) {
		const int v = tid >= d ? scan[tid - d] : 0;
		__syncthreads();
		scan[tid] += v;
		__syncthreads();
	}
	const int start = scan[tid] - cnt;
	if ( tid < E ) {
		ex_start[tid] = start;
		if ( cnt > 0 && p.heat != nullptr ) atomicAdd( &p.heat[tid], static_cast<uint32_t>( cnt ) );
		int pos = start;
		for ( int s = 0; s < n; ++s ) {
			if ( loc[s] == tid ) {
				order[pos] = s;
				inv[s] = pos;
				++pos;
			}
		}
	}
	if ( tid == 0 ) ex_start[E] = scan[GRP_THREADS - 1];
}

// One wave's run table: runs[0].x = number of runs, runs[1 + r] = (local expert, first position of the wave, rows)
__global__ __launch_bounds__( GRP_THREADS ) void grp_runs_kernel( const MoeCoopParams p, int E, const int* __restrict__ order, const int* __restrict__ ex_start, int4* __restrict__ runs, int P0, int W, int MR ) {
	__shared__ int scan[GRP_THREADS];
	const int tid = threadIdx.x;
	const int n_active = ex_start[E];
	const int Lw = max( 0, min( W, n_active - P0 ) );
	int flag = 0, e = 0, i = tid;
	if ( tid < Lw ) {
		const int slot = order[P0 + i];
		e = p.sel[slot] - p.min_expert;
		const int seg_start = max( P0, ex_start[e] );
		flag = ( ( P0 + i - seg_start ) % MR ) == 0 ? 1 : 0;
	}
	scan[tid] = flag;
	__syncthreads();
	for ( int d = 1; d < GRP_THREADS; d <<= 1 ) {
		const int v = tid >= d ? scan[tid - d] : 0;
		__syncthreads();
		scan[tid] += v;
		__syncthreads();
	}
	if ( tid < Lw && flag ) {
		const int seg_end = min( ex_start[e + 1], P0 + Lw );
		runs[1 + scan[tid] - 1] = make_int4( e, i, min( MR, seg_end - ( P0 + i ) ), 0 );
	}
	if ( tid == 0 ) runs[0] = make_int4( scan[GRP_THREADS - 1], Lw, 0, 0 );
}

// grid (W, H / 128, 2), 32 threads: position i of the wave
__global__ __launch_bounds__( 32 ) void grp_rot_kernel( const MoeCoopParams p, const int* __restrict__ order, const int4* __restrict__ runs, int P0, half* __restrict__ had_g, half* __restrict__ had_u ) {
	const int i = blockIdx.x, chunk = blockIdx.y, proj = blockIdx.z, lane = threadIdx.x;
	if ( i >= runs[0].y ) return;
	const int slot = order[P0 + i];
	const int local = p.sel[slot] - p.min_expert;
	const int row = slot / p.topk;
	const int col = chunk * 128 + lane * 4;
	const half* suh = ( proj ? p.u_suh : p.g_suh )[local];
	const uint2 xr = *reinterpret_cast<const uint2*>( p.x + static_cast<size_t>( row ) * p.H + col );
	const uint2 sr = *reinterpret_cast<const uint2*>( suh + col );
	const half2 a = __hmul2( __builtin_bit_cast( half2, xr.x ), __builtin_bit_cast( half2, sr.x ) );
	const half2 b = __hmul2( __builtin_bit_cast( half2, xr.y ), __builtin_bit_cast( half2, sr.y ) );
	float v0 = __low2float( a ), v1 = __high2float( a ), v2 = __low2float( b ), v3 = __high2float( b );
	had128( v0, v1, v2, v3, lane );
	store4( ( proj ? had_u : had_g ) + static_cast<size_t>( i ) * p.H + col, v0, v1, v2, v3 );
}

// A run's GEMV tile: up to MR rows share one decode of the expert's 16-column tile. grid (N / 16, W, projections); dynamic smem: the run's rows
// (MR * K halves) then red[warps][MR][16]
template <int bits, int MR, bool STAGED>
__global__ __launch_bounds__( 32 * GEMV_WARPS ) void grp_gemv_kernel(
	const uint16_t* const* __restrict__ trellis0,
	const uint16_t* const* __restrict__ trellis1,
	const half* __restrict__ in0,
	const half* __restrict__ in1,
	float* __restrict__ out0,
	float* __restrict__ out1,
	const int4* __restrict__ runs,
	int K,
	int N
) {
	extern __shared__ __align__( 16 ) unsigned char grp_smem[];
	half* sa = reinterpret_cast<half*>( grp_smem );
	// STAGED: the tile column's whole weight (k_tiles pieces of 32 * bits bytes) is read into shared memory first, every load issued at once,
	// so a cold expert's PCIe reads are all in flight together (hundreds of KB across the resident blocks) and the decode then runs from shared memory
	uint4* sw = reinterpret_cast<uint4*>( grp_smem + static_cast<size_t>( MR ) * K * sizeof( half ) );
	float* red = reinterpret_cast<float*>( grp_smem + static_cast<size_t>( MR ) * K * sizeof( half ) + ( STAGED ? static_cast<size_t>( K >> 4 ) * 32 * bits : 0 ) );
	const int r = blockIdx.y;
	if ( r >= runs[0].x ) return;
	const int4 run = runs[1 + r];
	const int local = run.x, i0 = run.y, m = run.z;
	const bool second = blockIdx.z != 0;
	const uint16_t* B = ( second ? trellis1 : trellis0 )[local];
	const half* A = ( second ? in1 : in0 ) + static_cast<size_t>( i0 ) * K;
	float* C = ( second ? out1 : out0 ) + static_cast<size_t>( i0 ) * N;
	const int tid = threadIdx.x, lane = tid & 31;
	const int warp = __builtin_amdgcn_readfirstlane( tid >> 5 );
	const int kc = K >> 3;
	const int tile_n = blockIdx.x, n_tiles = N >> 4, k_tiles = K >> 4;
	if constexpr ( STAGED ) {
		constexpr int Q = 2 * bits;                       // uint4 per k tile
		constexpr int ITEMS = 8;
		uint4 wv[ITEMS];
		const int total = k_tiles * Q;
		#pragma unroll
		for ( int i = 0; i < ITEMS; ++i ) {
			const int idx = tid + i * 32 * GEMV_WARPS;
			if ( idx < total ) {
				const int kt = idx / Q, q = idx - kt * Q;
				wv[i] = reinterpret_cast<const uint4*>( B + ( static_cast<size_t>( kt ) * n_tiles + tile_n ) * ( 16 * bits ) )[q];
			}
		}
		#pragma unroll
		for ( int i = 0; i < ITEMS; ++i ) {
			const int idx = tid + i * 32 * GEMV_WARPS;
			if ( idx < total ) sw[idx] = wv[i];
		}
	}
	const int per = ( k_tiles + GEMV_WARPS - 1 ) / GEMV_WARPS;
	const int kb0 = warp * per, kb1 = min( k_tiles, kb0 + per );
	const half2* ar = reinterpret_cast<const half2*>( sa ) + ( lane & 3 );
	// A run holds up to RUN_ROWS rows (2 * MR when staged: the weight column is in shared memory once and the rows go through it MR at a time)
	for ( int row0 = 0; row0 < m; row0 += MR ) {
		const int mg = min( MR, m - row0 );
		if ( row0 > 0 ) __syncthreads();   // the previous group is done with the rows and the reduction buffer
		for ( int idx = tid; idx < mg * kc; idx += 32 * GEMV_WARPS ) {
			const int j = idx / kc, c = idx - j * kc;
			reinterpret_cast<uint4*>( sa + static_cast<size_t>( j ) * K )[c] = reinterpret_cast<const uint4*>( A + static_cast<size_t>( row0 + j ) * K )[c];
		}
		__syncthreads();
		float acc_a[MR], acc_b[MR];
		#pragma unroll
		for ( int j = 0; j < MR; ++j ) acc_a[j] = acc_b[j] = 0.f;
		const uint32_t* bt = STAGED ? reinterpret_cast<const uint32_t*>( sw ) + static_cast<size_t>( kb0 ) * ( 8 * bits )
			: reinterpret_cast<const uint32_t*>( B + ( static_cast<size_t>( kb0 ) * n_tiles + tile_n ) * ( 16 * bits ) );
		const size_t bstep = STAGED ? static_cast<size_t>( 8 * bits ) : static_cast<size_t>( n_tiles ) * 16 * bits / 2;
		for ( int kt = kb0; kt < kb1; ++kt, bt += bstep ) {
			FragB f0, f1;
			dq_dispatch<bits, 2>( bt, lane << 3, f0, f1 );
			#pragma unroll
			for ( int j = 0; j < MR; ++j ) {
				if ( j < mg ) {
					const half2 a01 = ar[j * ( K / 2 ) + kt * 8], a89 = ar[j * ( K / 2 ) + kt * 8 + 4];
					acc_a[j] = fdot2( a01, f0[0], acc_a[j] );
					acc_a[j] = fdot2( a89, f0[1], acc_a[j] );
					acc_b[j] = fdot2( a01, f1[0], acc_b[j] );
					acc_b[j] = fdot2( a89, f1[1], acc_b[j] );
				}
			}
		}
		#pragma unroll
		for ( int j = 0; j < MR; ++j ) {
			float a = acc_a[j], b = acc_b[j];
			a += __shfl_xor( a, 1 );
			a += __shfl_xor( a, 2 );
			b += __shfl_xor( b, 1 );
			b += __shfl_xor( b, 2 );
			if ( ( lane & 3 ) == 0 ) {
				red[( warp * MR + j ) * 16 + ( lane >> 2 )] = a;
				red[( warp * MR + j ) * 16 + ( lane >> 2 ) + 8] = b;
			}
		}
		__syncthreads();
		if ( tid < mg * 16 ) {
			const int j = tid >> 4, c = tid & 15;
			float v = 0.f;
			#pragma unroll
			for ( int w = 0; w < GEMV_WARPS; ++w ) v += red[( w * MR + j ) * 16 + c];
			C[static_cast<size_t>( row0 + j ) * N + tile_n * 16 + c] = v;
		}
	}
}

// The same GEMV tile on the native WMMA: a run's rows (up to 16) are the A operand, and each k tile of the expert's weight tile is decoded
// (dq_dispatch: lane L holds W[k = 2 (L & 3) + {0, 1, 8, 9}][n = L >> 2, + 8]) into a 16 x 16 FP16 slab of the warp's own shared memory,
// whose rows (n, k contiguous) are the B operand: one WMMA replaces 32 quarter-rate dot2 per lane. The rows are staged through shared memory
// 512 k at a time. Block warps take the k tiles in turn; their accumulators are summed through shared memory at the end.
constexpr int WG_KC = 512;               // k per row chunk
constexpr int WG_SLAB = 24;              // halves per slab row (16 + 8: b128 reads of 8 rows hit distinct banks)
constexpr int WG_AROW = WG_KC + 8;       // halves per staged row

template <int bits, bool STAGED>
__global__ __launch_bounds__( 32 * GEMV_WARPS ) void grp_wmma_kernel(
	const uint16_t* const* __restrict__ trellis0,
	const uint16_t* const* __restrict__ trellis1,
	const half* __restrict__ in0,
	const half* __restrict__ in1,
	float* __restrict__ out0,
	float* __restrict__ out1,
	const int4* __restrict__ runs,
	int K,
	int N
) {
	extern __shared__ __align__( 16 ) unsigned char wg_smem[];
	half* sa = reinterpret_cast<half*>( wg_smem );                                                     // [16][WG_AROW]
	half* slab = sa + 16 * WG_AROW;                                                                   // [warp][16][WG_SLAB]
	float* red = reinterpret_cast<float*>( slab + GEMV_WARPS * 16 * WG_SLAB );                         // [warp][16][16]
	uint4* sw = reinterpret_cast<uint4*>( red + GEMV_WARPS * 256 );                                    // STAGED: the weight column
	const int r = blockIdx.y;
	if ( r >= runs[0].x ) return;
	const int4 run = runs[1 + r];
	const int local = run.x, i0 = run.y, m = run.z;
	const bool second = blockIdx.z != 0;
	const uint16_t* B = ( second ? trellis1 : trellis0 )[local];
	const half* A = ( second ? in1 : in0 ) + static_cast<size_t>( i0 ) * K;
	float* C = ( second ? out1 : out0 ) + static_cast<size_t>( i0 ) * N;
	const int tid = threadIdx.x, lane = tid & 31;
	const int warp = __builtin_amdgcn_readfirstlane( tid >> 5 );
	const int tile_n = blockIdx.x, n_tiles = N >> 4, k_tiles = K >> 4;
	if constexpr ( STAGED ) {
		constexpr int Q = 2 * bits;
		constexpr int ITEMS = 8;
		uint4 wv[ITEMS];
		const int total = k_tiles * Q;
		#pragma unroll
		for ( int i = 0; i < ITEMS; ++i ) {
			const int idx = tid + i * 32 * GEMV_WARPS;
			if ( idx < total ) {
				const int kt = idx / Q, q = idx - kt * Q;
				wv[i] = reinterpret_cast<const uint4*>( B + ( static_cast<size_t>( kt ) * n_tiles + tile_n ) * ( 16 * bits ) )[q];
			}
		}
		#pragma unroll
		for ( int i = 0; i < ITEMS; ++i ) {
			const int idx = tid + i * 32 * GEMV_WARPS;
			if ( idx < total ) sw[idx] = wv[i];
		}
	}
	const int ql = lane & 15, hi = lane >> 4, c8 = lane >> 2, r0 = ( lane & 3 ) * 2;
	half* my_slab = slab + warp * 16 * WG_SLAB;
	wm_v8f d = wm_v8f{ 0, 0, 0, 0, 0, 0, 0, 0 };
	for ( int k0 = 0; k0 < K; k0 += WG_KC ) {
		const int len = min( WG_KC, K - k0 );
		__syncthreads();   // the previous chunk's rows are no longer read
		for ( int idx = tid; idx < m * ( len >> 3 ); idx += 32 * GEMV_WARPS ) {
			const int j = idx / ( len >> 3 ), c = idx - j * ( len >> 3 );
			*reinterpret_cast<uint4*>( sa + static_cast<size_t>( j ) * WG_AROW + c * 8 ) = *reinterpret_cast<const uint4*>( A + static_cast<size_t>( j ) * K + k0 + c * 8 );
		}
		__syncthreads();
		for ( int kl = warp; kl < ( len >> 4 ); kl += GEMV_WARPS ) {
			const int kt = ( k0 >> 4 ) + kl;
			const uint32_t* bt = STAGED ? reinterpret_cast<const uint32_t*>( sw ) + static_cast<size_t>( kt ) * ( 8 * bits )
				: reinterpret_cast<const uint32_t*>( B + ( static_cast<size_t>( kt ) * n_tiles + tile_n ) * ( 16 * bits ) );
			FragB f0, f1;
			dq_dispatch<bits, 2>( bt, lane << 3, f0, f1 );
			*reinterpret_cast<uint32_t*>( my_slab + c8 * WG_SLAB + r0 ) = __builtin_bit_cast( uint32_t, f0[0] );
			*reinterpret_cast<uint32_t*>( my_slab + c8 * WG_SLAB + r0 + 8 ) = __builtin_bit_cast( uint32_t, f0[1] );
			*reinterpret_cast<uint32_t*>( my_slab + ( c8 + 8 ) * WG_SLAB + r0 ) = __builtin_bit_cast( uint32_t, f1[0] );
			*reinterpret_cast<uint32_t*>( my_slab + ( c8 + 8 ) * WG_SLAB + r0 + 8 ) = __builtin_bit_cast( uint32_t, f1[1] );
			__builtin_amdgcn_wave_barrier();
			asm volatile( "s_waitcnt lgkmcnt(0)" ::: "memory" );
			const wm_v16h b = wm_ld16( my_slab + ql * WG_SLAB );
			const wm_v16h a = wm_ld16( sa + static_cast<size_t>( ql ) * WG_AROW + kl * 16 );
			d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32( a, b, d );
			asm volatile( "s_waitcnt lgkmcnt(0)" ::: "memory" );
			__builtin_amdgcn_wave_barrier();
		}
	}
	#pragma unroll
	for ( int e = 0; e < 8; ++e ) red[warp * 256 + ( 2 * e + hi ) * 16 + ql] = d[e];
	__syncthreads();
	{
		const int j = tid >> 4, c = tid & 15;
		if ( j < m ) {
			float v = 0.f;
			#pragma unroll
			for ( int w = 0; w < GEMV_WARPS; ++w ) v += red[w * 256 + j * 16 + c];
			C[static_cast<size_t>( j ) * N + tile_n * 16 + c] = v;
		}
	}
}

template <int bits>
void grp_wmma_launch( const uint16_t* const* t0, const uint16_t* const* t1, const half* in0, const half* in1, float* out0, float* out1,
	const int4* runs, int K, int N, int W, int nproj, cudaStream_t stream ) {
	const size_t base = static_cast<size_t>( 16 ) * WG_AROW * sizeof( half ) + static_cast<size_t>( GEMV_WARPS ) * 16 * WG_SLAB * sizeof( half ) + static_cast<size_t>( GEMV_WARPS ) * 256 * sizeof( float );
	const size_t staged = base + static_cast<size_t>( K >> 4 ) * 32 * bits;
	const bool stage = staged <= 65000 && static_cast<size_t>( K >> 4 ) * 2 * bits <= 8 * 32 * GEMV_WARPS;
	if ( stage ) {
		CUDA_CHECK( cudaFuncSetAttribute( grp_wmma_kernel<bits, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( staged ) ) );
		grp_wmma_kernel<bits, true><<<dim3( N / 16, W, nproj ), 32 * GEMV_WARPS, staged, stream>>>( t0, t1, in0, in1, out0, out1, runs, K, N );
	}
	else {
		CUDA_CHECK( cudaFuncSetAttribute( grp_wmma_kernel<bits, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( base ) ) );
		grp_wmma_kernel<bits, false><<<dim3( N / 16, W, nproj ), 32 * GEMV_WARPS, base, stream>>>( t0, t1, in0, in1, out0, out1, runs, K, N );
	}
}

void grp_wmma_bits( int bits, const uint16_t* const* t0, const uint16_t* const* t1, const half* in0, const half* in1, float* out0, float* out1,
	const int4* runs, int K, int N, int W, int nproj, cudaStream_t stream ) {
	switch ( bits ) {
		case 1: grp_wmma_launch<1>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 2: grp_wmma_launch<2>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 3: grp_wmma_launch<3>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 4: grp_wmma_launch<4>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 5: grp_wmma_launch<5>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 6: grp_wmma_launch<6>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 7: grp_wmma_launch<7>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		default: grp_wmma_launch<8>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
	}
}

// grid (W, I / 128), 32 threads
__global__ __launch_bounds__( 32 ) void grp_mid_kernel( const MoeCoopParams p, const int* __restrict__ order, const int4* __restrict__ runs, int P0,
	const float* __restrict__ gu_g, const float* __restrict__ gu_u, half* __restrict__ act ) {
	const int i = blockIdx.x, chunk = blockIdx.y, lane = threadIdx.x;
	if ( i >= runs[0].y ) return;
	const int local = p.sel[order[P0 + i]] - p.min_expert;
	const int col = chunk * 128 + lane * 4;
	const size_t off = static_cast<size_t>( i ) * p.I + col;
	const float4 uf = *reinterpret_cast<const float4*>( gu_u + off );
	const float4 gf = *reinterpret_cast<const float4*>( gu_g + off );
	float u0 = uf.x, u1 = uf.y, u2 = uf.z, u3 = uf.w, g0 = gf.x, g1 = gf.y, g2 = gf.z, g3 = gf.w;
	had128( u0, u1, u2, u3, lane );
	scale4( p.u_svh[local] + col, u0, u1, u2, u3 );
	had128( g0, g1, g2, g3, lane );
	scale4( p.g_svh[local] + col, g0, g1, g2, g3 );
	float a0 = act_gate( g0, u0, p.act_limit ), a1 = act_gate( g1, u1, p.act_limit ), a2 = act_gate( g2, u2, p.act_limit ), a3 = act_gate( g3, u3, p.act_limit );
	scale4( p.d_suh[local] + col, a0, a1, a2, a3 );
	had128( a0, a1, a2, a3, lane );
	store4( act + off, a0, a1, a2, a3 );
}

// grid (rows, H / 128), 32 threads: adds this wave's slots of the row to acc, in ascending expert order
__global__ __launch_bounds__( 32 ) void grp_acc_kernel( const MoeCoopParams p, const int* __restrict__ inv, const int4* __restrict__ runs, int P0,
	const float* __restrict__ d_out, float* __restrict__ acc ) {
	const int row = blockIdx.x, chunk = blockIdx.y, lane = threadIdx.x;
	const int Lw = runs[0].y;
	int pe[MOE_COOP_TOPK], pp[MOE_COOP_TOPK], ps[MOE_COOP_TOPK], cnt = 0;
	for ( int k = 0; k < p.topk; ++k ) {
		const int s = row * p.topk + k;
		const int pos = inv[s];
		if ( pos >= P0 && pos < P0 + Lw ) {
			const int e = p.sel[s] - p.min_expert;
			int j = cnt++;
			while ( j > 0 && pe[j - 1] > e ) { pe[j] = pe[j - 1]; pp[j] = pp[j - 1]; ps[j] = ps[j - 1]; --j; }
			pe[j] = e; pp[j] = pos; ps[j] = s;
		}
	}
	if ( cnt == 0 ) return;
	const int col = chunk * 128 + lane * 4;
	float4 a = *reinterpret_cast<const float4*>( acc + static_cast<size_t>( row ) * p.H + col );
	for ( int j = 0; j < cnt; ++j ) {
		const float w = __half2float( p.rw[ps[j]] );
		const float4 d = *reinterpret_cast<const float4*>( d_out + static_cast<size_t>( pp[j] - P0 ) * p.H + col );
		float v0 = d.x, v1 = d.y, v2 = d.z, v3 = d.w;
		had128( v0, v1, v2, v3, lane );
		scale4( p.d_svh[pe[j]] + col, v0, v1, v2, v3 );
		a.x += w * v0; a.y += w * v1; a.z += w * v2; a.w += w * v3;
	}
	*reinterpret_cast<float4*>( acc + static_cast<size_t>( row ) * p.H + col ) = a;
}

// grid (rows, H / 128), 32 threads
__global__ __launch_bounds__( 32 ) void grp_fin_kernel( const MoeCoopParams p, const float* __restrict__ acc ) {
	const int row = blockIdx.x, col = blockIdx.y * 128 + threadIdx.x * 4;
	const float4 a = *reinterpret_cast<const float4*>( acc + static_cast<size_t>( row ) * p.H + col );
	store4( p.out + static_cast<size_t>( row ) * p.H + col, a.x, a.y, a.z, a.w );
}

template <int bits, int MR>
void grp_gemv_launch( const uint16_t* const* t0, const uint16_t* const* t1, const half* in0, const half* in1, float* out0, float* out1,
	const int4* runs, int K, int N, int W, int nproj, cudaStream_t stream ) {
	const size_t base = static_cast<size_t>( MR ) * K * sizeof( half ) + static_cast<size_t>( GEMV_WARPS ) * MR * 16 * sizeof( float );
	const size_t staged = base + static_cast<size_t>( K >> 4 ) * 32 * bits;
	// Staging needs the weight column (<= 8 uint4 per thread) and the shared memory for it next to the rows
	const bool stage = staged <= 65000 && static_cast<size_t>( K >> 4 ) * 2 * bits <= 8 * 32 * GEMV_WARPS;
	if ( stage ) {
		CUDA_CHECK( cudaFuncSetAttribute( grp_gemv_kernel<bits, MR, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( staged ) ) );
		grp_gemv_kernel<bits, MR, true><<<dim3( N / 16, W, nproj ), 32 * GEMV_WARPS, staged, stream>>>( t0, t1, in0, in1, out0, out1, runs, K, N );
	}
	else {
		CUDA_CHECK( cudaFuncSetAttribute( grp_gemv_kernel<bits, MR, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( base ) ) );
		grp_gemv_kernel<bits, MR, false><<<dim3( N / 16, W, nproj ), 32 * GEMV_WARPS, base, stream>>>( t0, t1, in0, in1, out0, out1, runs, K, N );
	}
}

template <int MR>
void grp_gemv_bits( int bits, const uint16_t* const* t0, const uint16_t* const* t1, const half* in0, const half* in1, float* out0, float* out1,
	const int4* runs, int K, int N, int W, int nproj, cudaStream_t stream ) {
	switch ( bits ) {
		case 1: grp_gemv_launch<1, MR>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 2: grp_gemv_launch<2, MR>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 3: grp_gemv_launch<3, MR>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 4: grp_gemv_launch<4, MR>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 5: grp_gemv_launch<5, MR>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 6: grp_gemv_launch<6, MR>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		case 7: grp_gemv_launch<7, MR>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
		default: grp_gemv_launch<8, MR>( t0, t1, in0, in1, out0, out1, runs, K, N, W, nproj, stream ); break;
	}
}

size_t align256( size_t v ) { return ( v + 255 ) & ~static_cast<size_t>( 255 ); }

} // namespace

bool exl3_moe_rocm_grouped_launch( const MoeCoopParams& p, cudaStream_t stream ) {
	const int n = p.bsz * p.topk;
	const int E = p.max_expert - p.min_expert;
	if ( n > GRP_MAX_SLOTS || E < 1 || E > GRP_THREADS - 1 || p.grp_ws == nullptr || p.topk > MOE_COOP_TOPK || p.H % 128 || p.I % 128 ) return false;
	// Run size: the rows of a run sit in shared memory next to the reduction buffer (the largest K is the hidden width)
	const int maxK = max( p.H, p.I );
	const int MR = static_cast<size_t>( 8 ) * maxK * sizeof( half ) + GEMV_WARPS * 8 * 16 * sizeof( float ) <= 60000 ? 8 : 4;
	// Carve the workspace: fixed parts, then per-position buffers for a wave of W positions
	const size_t fixed = align256( 2 * sizeof( int ) * n ) + align256( sizeof( int ) * ( E + 1 ) ) + align256( static_cast<size_t>( p.bsz ) * p.H * sizeof( float ) );
	const size_t per_pos = 2 * p.H * sizeof( half ) + 2 * p.I * sizeof( float ) + p.I * sizeof( half ) + p.H * sizeof( float ) + sizeof( int4 ) + 1024;
	if ( p.grp_ws_bytes <= fixed ) return false;
	int W = static_cast<int>( min( static_cast<size_t>( GRP_THREADS ), ( p.grp_ws_bytes - fixed ) / per_pos ) ) & ~7;
	if ( W < 64 ) return false;
	char* base = static_cast<char*>( p.grp_ws );
	size_t off = 0;
	auto take = [&]( size_t bytes ) { char* r = base + off; off += align256( bytes ); return r; };
	int* order = reinterpret_cast<int*>( take( sizeof( int ) * n ) );
	int* inv = reinterpret_cast<int*>( take( sizeof( int ) * n ) );
	int* ex_start = reinterpret_cast<int*>( take( sizeof( int ) * ( E + 1 ) ) );
	float* acc = reinterpret_cast<float*>( take( static_cast<size_t>( p.bsz ) * p.H * sizeof( float ) ) );
	int4* runs = reinterpret_cast<int4*>( take( sizeof( int4 ) * ( W + 1 ) ) );
	half* had_g = reinterpret_cast<half*>( take( static_cast<size_t>( W ) * p.H * sizeof( half ) ) );
	half* had_u = reinterpret_cast<half*>( take( static_cast<size_t>( W ) * p.H * sizeof( half ) ) );
	float* gu_g = reinterpret_cast<float*>( take( static_cast<size_t>( W ) * p.I * sizeof( float ) ) );
	float* gu_u = reinterpret_cast<float*>( take( static_cast<size_t>( W ) * p.I * sizeof( float ) ) );
	half* act = reinterpret_cast<half*>( take( static_cast<size_t>( W ) * p.I * sizeof( half ) ) );
	float* d_out = reinterpret_cast<float*>( take( static_cast<size_t>( W ) * p.H * sizeof( float ) ) );
	if ( off > p.grp_ws_bytes ) return false;

	static const bool use_wmma = !engine::runtime::options().moe_gemv;
	CUDA_CHECK( cudaMemsetAsync( acc, 0, static_cast<size_t>( p.bsz ) * p.H * sizeof( float ), stream ) );
	grp_group_kernel<<<1, GRP_THREADS, 0, stream>>>( p, E, order, inv, ex_start );
	const int waves = ( n + W - 1 ) / W;
	for ( int w = 0; w < waves; ++w ) {
		const int P0 = w * W;
		grp_runs_kernel<<<1, GRP_THREADS, 0, stream>>>( p, E, order, ex_start, runs, P0, W, 16 );   // runs of up to 16 rows, handled MR at a time
		grp_rot_kernel<<<dim3( W, p.H / 128, 2 ), 32, 0, stream>>>( p, order, runs, P0, had_g, had_u );
		if ( use_wmma ) grp_wmma_bits( p.bits, p.g_trellis, p.u_trellis, had_g, had_u, gu_g, gu_u, runs, p.H, p.I, W, 2, stream );
		else if ( MR == 8 ) grp_gemv_bits<8>( p.bits, p.g_trellis, p.u_trellis, had_g, had_u, gu_g, gu_u, runs, p.H, p.I, W, 2, stream );
		else grp_gemv_bits<4>( p.bits, p.g_trellis, p.u_trellis, had_g, had_u, gu_g, gu_u, runs, p.H, p.I, W, 2, stream );
		grp_mid_kernel<<<dim3( W, p.I / 128 ), 32, 0, stream>>>( p, order, runs, P0, gu_g, gu_u, act );
		if ( use_wmma ) grp_wmma_bits( p.bits, p.d_trellis, p.d_trellis, act, act, d_out, d_out, runs, p.I, p.H, W, 1, stream );
		else if ( MR == 8 ) grp_gemv_bits<8>( p.bits, p.d_trellis, p.d_trellis, act, act, d_out, d_out, runs, p.I, p.H, W, 1, stream );
		else grp_gemv_bits<4>( p.bits, p.d_trellis, p.d_trellis, act, act, d_out, d_out, runs, p.I, p.H, W, 1, stream );
		grp_acc_kernel<<<dim3( p.bsz, p.H / 128 ), 32, 0, stream>>>( p, inv, runs, P0, d_out, acc );
	}
	grp_fin_kernel<<<dim3( p.bsz, p.H / 128 ), 32, 0, stream>>>( p, acc );
	return true;
}
#endif
