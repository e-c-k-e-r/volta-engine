#include "engine/modules/nvfp4_moe.h"
#include "engine/modules/moe.h"
#include "nvfp4_common.cuh"
#include <cstdint>

// Fused NVFP4 routed-expert kernels: all of a layer's routes in three launches (dispatch, gate/up/SiLU, down + ordered weighted sum).
// Routes are grouped by expert: a block streams one expert's weight tile once and multiplies it with up to 8 of the rows routed to it
// (the 8 rows of the Volta m8n8k4 A operand, or the N of the swapped SM_80+ m16n8k16); experts with more rows loop over chunks of 8.
// Weight layout per matrix: tiles of 32 output columns, each [K/16 groups][32 lanes] of uint2 codes (+ one E4M3 scale byte per lane and group).
//
// Dispatch buffer (ints): [0] active experts, [DISP_HDR + e] routes of local expert e, [DISP_HDR + E + i] i-th active expert,
// [DISP_HDR + 2E + e * ROWS + j] j-th route (row * topk + k) of expert e

using engine::modules::NVFP4ExpertTable;

#define DISP_HDR 4
#define ROWS_CAP MOE_NVFP4_ROWS

__global__ __launch_bounds__( 1024 )
void nvfp4_moe_dispatch( const int* __restrict__ route_idx, int n_routes, int min_expert, int num_local, uint32_t* __restrict__ heat, int* __restrict__ disp ) {
	__shared__ int cnt[MOE_MAX_EXPERTS];
	__shared__ int nact;
	for ( int e = threadIdx.x; e < num_local; e += blockDim.x ) cnt[e] = 0;
	if ( threadIdx.x == 0 ) nact = 0;
	__syncthreads();
	int* list = disp + DISP_HDR + 2 * num_local;
	for ( int r = threadIdx.x; r < n_routes; r += blockDim.x ) {
		const int e = route_idx[r] - min_expert;
		if ( ( unsigned ) e < ( unsigned ) num_local ) {
			const int pos = atomicAdd( &cnt[e], 1 );
			list[e * ROWS_CAP + pos] = r;
		}
	}
	__syncthreads();
	for ( int e = threadIdx.x; e < num_local; e += blockDim.x ) {
		const int c = cnt[e];
		disp[DISP_HDR + e] = c;
		if ( c > 0 ) {
			disp[DISP_HDR + num_local + atomicAdd( &nact, 1 )] = e;
			if ( heat != nullptr ) atomicAdd( heat + e, ( uint32_t ) c );
		}
	}
	__syncthreads();
	if ( threadIdx.x == 0 ) disp[0] = nact;
}

// Row 0..7 accumulator element (register i of lane) -> (row, col within the 32-column tile)
DEV_INLINE int acc_row( int i, int lane ) { return nvfp4_acc_row( i, lane ); }
DEV_INLINE int acc_col( int i, int lane ) { return nvfp4_acc_col( i, lane ); }

// Gate and up of one 32-column tile for the rows routed to one expert; epilogue silu(gate) * up -> h[route][col]
template <int WARPS, bool GROUPED>
__global__ __launch_bounds__( WARPS * 32 )
void nvfp4_moe_gate_up( NVFP4ExpertTable t, const half* __restrict__ x, const int* __restrict__ disp, const int* __restrict__ route_idx, uint32_t* __restrict__ heat,
	int topk, float act_limit, half* __restrict__ h ) {
	// Ungrouped (one row): block row y is route y, no dispatch pass
	const int E = t.num_local;
	int e, cnt;
	const int* list;
	if ( GROUPED ) {
		if ( ( int ) blockIdx.y >= disp[0] ) return;
		e = disp[DISP_HDR + E + blockIdx.y];
		cnt = disp[DISP_HDR + e];
		list = disp + DISP_HDR + 2 * E + e * ROWS_CAP;
	}
	else {
		e = route_idx[blockIdx.y] - t.min_expert;
		if ( ( unsigned ) e >= ( unsigned ) E ) return;
		if ( heat != nullptr && blockIdx.x == 0 && threadIdx.x == 0 ) atomicAdd( heat + e, 1u );
		cnt = 1;
		list = &route_idx[0];   // unused
	}
	#define ROUTE_OF( j ) ( GROUPED ? list[j] : ( int ) blockIdx.y )
	extern __shared__ __align__( 16 ) unsigned char smem[];   // the chunk's input rows [8][K] halves, then (after the loop) the per-warp partial sums
	half* sx = reinterpret_cast<half*>( smem );
	float* cs = reinterpret_cast<float*>( smem );   // [warp][gate / up][rows * 32] (one row when ungrouped)
	constexpr int CS = GROUPED ? 8 * 32 : 32;
	const int K = t.hidden, I = t.inter;
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int tile = blockIdx.x;
	const int G = K >> 4, Gq = G / WARPS, g0 = warp * Gq;
	const int r = nvfp4_x_row( lane );
	const size_t off = ( size_t ) tile * G * 32 + lane;
	const uint2* gc = reinterpret_cast<const uint2*>( t.gate_c[e] ) + off;
	const uint2* uc = reinterpret_cast<const uint2*>( t.up_c[e] ) + off;
	const uint8_t* gs = t.gate_s[e] + off;
	const uint8_t* us = t.up_s[e] + off;
	const half2 gmg = __float2half2_rn( t.gate_g[e] * 16384.f );
	const half2 gmu = __float2half2_rn( t.up_g[e] * 16384.f );

	const int CH = nvfp4_moe_chunk_rows( K );
	for ( int c0 = 0; c0 < cnt; c0 += CH ) {
		const int m = min( CH, cnt - c0 );
		for ( int i = threadIdx.x; i < m * ( K / 8 ); i += WARPS * 32 ) {
			const int j = i / ( K / 8 ), c = i % ( K / 8 );
			*reinterpret_cast<uint4*>( sx + ( size_t ) j * K + c * 8 ) = __ldg( reinterpret_cast<const uint4*>( x + ( size_t ) ( ROUTE_OF( c0 + j ) / topk ) * K ) + c );
		}
		__syncthreads();
		const half* xr = r < m ? sx + ( size_t ) r * K : nullptr;
		float cg[8], cu[8];
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) { cg[i] = 0.f; cu[i] = 0.f; }

#if NVFP4_MMA_SM80
		#pragma unroll 5
		for ( int g = g0; g < g0 + Gq; g++ ) {
			unsigned aw[2][4], bx[2];
			nvfp4_xfrag( xr != nullptr ? xr + g * 16 : nullptr, bx );
			nvfp4_wfrag<true>( gc + ( size_t ) g * 32, gs + ( size_t ) g * 32, gmg, aw );
			nvfp4_step( cg, aw, bx );
			nvfp4_wfrag<true>( uc + ( size_t ) g * 32, us + ( size_t ) g * 32, gmu, aw );
			nvfp4_step( cu, aw, bx );
		}
#else
		#pragma unroll 5
		for ( int g = g0; g < g0 + Gq; g++ ) {
			const uint2 qg = __ldcs( gc + ( size_t ) g * 32 );
			const uint2 qu = __ldcs( uc + ( size_t ) g * 32 );
			const half2 scg = __hmul2( fp8e4m3_to_half2( __ldcs( gs + ( size_t ) g * 32 ) ), gmg );
			const half2 scu = __hmul2( fp8e4m3_to_half2( __ldcs( us + ( size_t ) g * 32 ) ), gmu );
			uint4 a01 = make_uint4( 0, 0, 0, 0 ), a23 = make_uint4( 0, 0, 0, 0 );
			if ( xr != nullptr ) {
				a01 = *reinterpret_cast<const uint4*>( xr + g * 16 );
				a23 = *reinterpret_cast<const uint4*>( xr + g * 16 + 8 );
			}
			const unsigned* A0 = reinterpret_cast<const unsigned*>( &a01 );
			const unsigned* A1 = reinterpret_cast<const unsigned*>( &a23 );
			half2 b[8];
			dequant8_tm( qg.x, scg, b + 0 );
			dequant8_tm( qg.y, scg, b + 4 );
			const unsigned* B = reinterpret_cast<const unsigned*>( b );
			MMA_8N8K4( cg, A0[0], A0[1], B[0], B[1] );
			MMA_8N8K4( cg, A0[2], A0[3], B[2], B[3] );
			MMA_8N8K4( cg, A1[0], A1[1], B[4], B[5] );
			MMA_8N8K4( cg, A1[2], A1[3], B[6], B[7] );
			dequant8_tm( qu.x, scu, b + 0 );
			dequant8_tm( qu.y, scu, b + 4 );
			MMA_8N8K4( cu, A0[0], A0[1], B[0], B[1] );
			MMA_8N8K4( cu, A0[2], A0[3], B[2], B[3] );
			MMA_8N8K4( cu, A1[0], A1[1], B[4], B[5] );
			MMA_8N8K4( cu, A1[2], A1[3], B[6], B[7] );
		}
#endif

		__syncthreads();   // every warp is done reading the input rows
		if ( GROUPED ) {
			#pragma unroll
			for ( int i = 0; i < 8; i++ ) {
				cs[( warp * 2 + 0 ) * CS + acc_row( i, lane ) * 32 + acc_col( i, lane )] = cg[i];
				cs[( warp * 2 + 1 ) * CS + acc_row( i, lane ) * 32 + acc_col( i, lane )] = cu[i];
			}
		}
#if NVFP4_MMA_SM80
		else if ( ( lane & 3 ) == 0 ) {   // row 0 only
			#pragma unroll
			for ( int i = 0; i < 8; i += 2 ) {   // registers 0, 2, 4, 6
				cs[( warp * 2 + 0 ) * CS + acc_col( i, lane )] = cg[i];
				cs[( warp * 2 + 1 ) * CS + acc_col( i, lane )] = cu[i];
			}
		}
#else
		else if ( ( lane & 17 ) == 0 ) {   // row 0 only
			#pragma unroll
			for ( int i = 0; i < 8; i += ( i & 1 ) ? 3 : 1 ) {   // registers 0, 1, 4, 5
				cs[( warp * 2 + 0 ) * CS + acc_col( i, lane )] = cg[i];
				cs[( warp * 2 + 1 ) * CS + acc_col( i, lane )] = cu[i];
			}
		}
#endif
		__syncthreads();
		if ( threadIdx.x < 256 && ( threadIdx.x >> 5 ) < m ) {
			float g = 0.f, u = 0.f;
			#pragma unroll
			for ( int w = 0; w < WARPS; w++ ) { g += cs[( w * 2 + 0 ) * CS + threadIdx.x]; u += cs[( w * 2 + 1 ) * CS + threadIdx.x]; }
			const float gh = __half2float( __float2half( g ) ), uh = __half2float( __float2half( u ) );
			// act_limit > 0 (GLM): up clamped both ways, the activated gate from above (as exl3_moe_coop's act_gate)
		float act = gh / ( 1.0f + __expf( -gh ) ), up = uh;
		if ( act_limit > 0.0f ) { up = fminf( fmaxf( up, -act_limit ), act_limit ); act = fminf( act, act_limit ); }
		h[( size_t ) ROUTE_OF( c0 + ( threadIdx.x >> 5 ) ) * I + tile * 32 + ( threadIdx.x & 31 )] = __float2half( act * up );
		}
		__syncthreads();
	}
}

// Down projection of one expert's rows for one 32-column tile into partial[route][col]; the last block to finish a tile sums every row's routes
// in order, weighted
template <int WARPS, bool GROUPED>
__global__ __launch_bounds__( WARPS * 32 )
void nvfp4_moe_down( NVFP4ExpertTable t, const half* __restrict__ h, const int* __restrict__ disp, const int* __restrict__ route_idx,
	const half* __restrict__ route_w, int rows, int topk, float* __restrict__ partial, int* __restrict__ counters, half* __restrict__ out ) {
	// Grouped: block row y is the y-th active expert; with no local route block 0 of each tile still writes the zero output.
	// Ungrouped (one row): block row y is route y, every route's block takes part in the tile's completion count
	const int E = t.num_local;
	int nact, e, cnt;
	const int* list;
	if ( GROUPED ) {
		nact = max( disp[0], 1 );
		if ( ( int ) blockIdx.y >= nact ) return;
		e = disp[0] > 0 ? disp[DISP_HDR + E + blockIdx.y] : 0;
		cnt = disp[0] > 0 ? disp[DISP_HDR + e] : 0;
		list = disp + DISP_HDR + 2 * E + e * ROWS_CAP;
	}
	else {
		nact = topk;
		e = route_idx[blockIdx.y] - t.min_expert;
		cnt = ( unsigned ) e < ( unsigned ) E ? 1 : 0;
		list = route_idx;   // unused
	}
	const int H = t.hidden, K = t.inter;
	const int tile = blockIdx.x;
	extern __shared__ __align__( 16 ) unsigned char smem[];   // the chunk's input rows [8][K] halves, then the per-warp partial sums
	half* sx = reinterpret_cast<half*>( smem );
	float* cs = reinterpret_cast<float*>( smem );
	constexpr int CS = GROUPED ? 8 * 32 : 32;
	__shared__ int last;
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int G = K >> 4, Gq = G / WARPS, g0 = warp * Gq;
	const int r = nvfp4_x_row( lane );
	const size_t off = ( size_t ) tile * G * 32 + lane;
	const uint2* dc = reinterpret_cast<const uint2*>( t.down_c[e] ) + off;
	const uint8_t* ds = t.down_s[e] + off;
	const half2 gm = __float2half2_rn( t.down_g[e] * 16384.f );

	const int CH = nvfp4_moe_chunk_rows( K );
	for ( int c0 = 0; c0 < cnt; c0 += CH ) {
		const int m = min( CH, cnt - c0 );
		for ( int i = threadIdx.x; i < m * ( K / 8 ); i += WARPS * 32 ) {
			const int j = i / ( K / 8 ), c = i % ( K / 8 );
			*reinterpret_cast<uint4*>( sx + ( size_t ) j * K + c * 8 ) = *( reinterpret_cast<const uint4*>( h + ( size_t ) ROUTE_OF( c0 + j ) * K ) + c );
		}
		__syncthreads();
		const half* hr = r < m ? sx + ( size_t ) r * K : nullptr;
		float c[8];
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) c[i] = 0.f;
#if NVFP4_MMA_SM80
		#pragma unroll
		for ( int g = g0; g < g0 + Gq; g++ ) {
			unsigned aw[2][4], bx[2];
			nvfp4_xfrag( hr != nullptr ? hr + g * 16 : nullptr, bx );
			nvfp4_wfrag<true>( dc + ( size_t ) g * 32, ds + ( size_t ) g * 32, gm, aw );
			nvfp4_step( c, aw, bx );
		}
#else
		#pragma unroll
		for ( int g = g0; g < g0 + Gq; g++ ) {
			const uint2 q = __ldcs( dc + ( size_t ) g * 32 );
			const half2 sc = __hmul2( fp8e4m3_to_half2( __ldcs( ds + ( size_t ) g * 32 ) ), gm );
			uint4 a01 = make_uint4( 0, 0, 0, 0 ), a23 = make_uint4( 0, 0, 0, 0 );
			if ( hr != nullptr ) {
				a01 = *reinterpret_cast<const uint4*>( hr + g * 16 );
				a23 = *reinterpret_cast<const uint4*>( hr + g * 16 + 8 );
			}
			const unsigned* A0 = reinterpret_cast<const unsigned*>( &a01 );
			const unsigned* A1 = reinterpret_cast<const unsigned*>( &a23 );
			half2 b[8];
			dequant8_tm( q.x, sc, b + 0 );
			dequant8_tm( q.y, sc, b + 4 );
			const unsigned* B = reinterpret_cast<const unsigned*>( b );
			MMA_8N8K4( c, A0[0], A0[1], B[0], B[1] );
			MMA_8N8K4( c, A0[2], A0[3], B[2], B[3] );
			MMA_8N8K4( c, A1[0], A1[1], B[4], B[5] );
			MMA_8N8K4( c, A1[2], A1[3], B[6], B[7] );
		}
#endif
		__syncthreads();
		if ( GROUPED ) {
			#pragma unroll
			for ( int i = 0; i < 8; i++ ) cs[warp * CS + acc_row( i, lane ) * 32 + acc_col( i, lane )] = c[i];
		}
#if NVFP4_MMA_SM80
		else if ( ( lane & 3 ) == 0 ) {   // row 0 only: registers 0, 2, 4, 6
			#pragma unroll
			for ( int i = 0; i < 8; i += 2 ) cs[warp * CS + acc_col( i, lane )] = c[i];
		}
#else
		else if ( ( lane & 17 ) == 0 ) {   // row 0 only
			#pragma unroll
			for ( int i = 0; i < 8; i += ( i & 1 ) ? 3 : 1 ) cs[warp * CS + acc_col( i, lane )] = c[i];
		}
#endif
		__syncthreads();
		if ( threadIdx.x < 256 && ( threadIdx.x >> 5 ) < m ) {
			float v = 0.f;
			#pragma unroll
			for ( int w = 0; w < WARPS; w++ ) v += cs[w * CS + threadIdx.x];
			partial[( size_t ) ROUTE_OF( c0 + ( threadIdx.x >> 5 ) ) * H + tile * 32 + ( threadIdx.x & 31 )] = __half2float( __float2half( v ) );
		}
		__syncthreads();
	}
	// Last block of the tile reduces over routes (fixed order)
	__threadfence();
	__syncthreads();
	if ( threadIdx.x == 0 ) last = atomicAdd( counters + tile, 1 ) == nact - 1;
	__syncthreads();
	if ( last ) {
		__threadfence();
		for ( int o = threadIdx.x; o < rows * 32; o += blockDim.x ) {
			const int row = o >> 5, col = o & 31;
			float acc = 0.f;
			for ( int k = 0; k < topk; k++ ) {
				const int rt = row * topk + k;
				if ( ( unsigned ) ( route_idx[rt] - t.min_expert ) < ( unsigned ) E ) acc += __half2float( route_w[rt] ) * __ldcg( partial + ( size_t ) rt * H + tile * 32 + col );
			}
			out[( size_t ) row * H + tile * 32 + col] = __float2half( acc );
		}
		if ( threadIdx.x == 0 ) counters[tile] = 0;
	}
}

namespace engine::modules {

void NVFP4MoE::forward_rows( const NVFP4ExpertTable& t, const half* x, const int* route_idx, const half* route_w, int rows, int topk,
	half* h, float* partial, int* disp, int* counters, uint32_t* heat, float act_limit, half* out, backend::stream_t stream ) {
	ENGINE_ASSERT( t.hidden % 64 == 0 && t.inter % 64 == 0 && t.inter % 32 == 0 && t.hidden % 32 == 0, "NVFP4 MoE: dims" );
	ENGINE_ASSERT( ( t.hidden / 16 ) % 16 == 0 && ( t.inter / 16 ) % 8 == 0 && rows <= ROWS_CAP && t.num_local <= MOE_MAX_EXPERTS, "NVFP4 MoE: group counts / rows" );
	// Up to 8 staged rows of K halves: 64 KB at GLM's K = 4096, above the 48 KB default
	static bool opted[ENGINE_NUM_GPUS] = {};
	int device = 0;
	CUDA_CHECK( cudaGetDevice( &device ) );
	if ( !opted[device] ) {
		CUDA_CHECK( cudaFuncSetAttribute( nvfp4_moe_gate_up<16, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 ) );
		CUDA_CHECK( cudaFuncSetAttribute( nvfp4_moe_gate_up<16, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304 ) );
		CUDA_CHECK( cudaFuncSetAttribute( nvfp4_moe_down<8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 ) );
		opted[device] = true;
	}
	if ( rows == 1 ) {
		nvfp4_moe_gate_up<16, false><<<dim3( t.inter / 32, topk ), 512, MAX( t.hidden * 2, 16 * 2 * 32 * 4 ), stream>>>( t, x, disp, route_idx, heat, topk, act_limit, h );
		nvfp4_moe_down<8, false><<<dim3( t.hidden / 32, topk ), 256, MAX( t.inter * 2, 8 * 32 * 4 ), stream>>>( t, h, disp, route_idx, route_w, rows, topk, partial, counters, out );
		return;
	}
	const int active_max = MIN( t.num_local, rows * topk );
	nvfp4_moe_dispatch<<<1, 1024, 0, stream>>>( route_idx, rows * topk, t.min_expert, t.num_local, heat, disp );
	nvfp4_moe_gate_up<16, true><<<dim3( t.inter / 32, active_max ), 512, MAX( MIN( rows, nvfp4_moe_chunk_rows( t.hidden ) ) * t.hidden * 2, 16 * 2 * 8 * 32 * 4 ), stream>>>( t, x, disp, route_idx, nullptr, topk, act_limit, h );
	nvfp4_moe_down<8, true><<<dim3( t.hidden / 32, active_max ), 256, MAX( MIN( rows, nvfp4_moe_chunk_rows( t.inter ) ) * t.inter * 2, 8 * 8 * 32 * 4 ), stream>>>( t, h, disp, route_idx, route_w, rows, topk, partial, counters, out );
}

} // namespace engine::modules
