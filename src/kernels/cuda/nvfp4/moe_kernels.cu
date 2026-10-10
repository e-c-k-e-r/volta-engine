#include "engine/modules/nvfp4_moe.h"
#include "nvfp4_common.cuh"
#include <cstdint>

// Fused NVFP4 routed-expert kernels for decode (one row): all of a layer's routes in two launches.
// Weight layout per matrix: tiles of 32 output columns, each [K/16 groups][32 lanes] of uint2 codes (+ one E4M3 scale byte per lane and group).

// Gate and up for one 32-column tile of one route, epilogue silu(gate) * up -> h[route][col]
template <int WARPS>
__global__ __launch_bounds__( WARPS * 32 )
void nvfp4_moe_gate_up( engine::modules::NVFP4ExpertTable t, const half* __restrict__ x, const int* __restrict__ route_idx, half* __restrict__ h ) {
	const int route = blockIdx.y;
	const int e = route_idx[route];
	if ( e >= t.num_local ) return;
	extern __shared__ half sx[];   // K halves
	__shared__ float cs[WARPS][2][32];
	const int K = t.hidden, I = t.inter;
	for ( int i = threadIdx.x * 8; i < K; i += WARPS * 32 * 8 ) *reinterpret_cast<uint4*>( sx + i ) = *reinterpret_cast<const uint4*>( x + i );
	__syncthreads();

	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	const int tile = blockIdx.x;
	const int G = K >> 4, Gq = G / WARPS, g0 = warp * Gq;
	const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
	const size_t off = ( size_t ) tile * G * 32 + lane;
	const uint2* gc = reinterpret_cast<const uint2*>( t.gate_c[e] ) + off;
	const uint2* uc = reinterpret_cast<const uint2*>( t.up_c[e] ) + off;
	const uint8_t* gs = t.gate_s[e] + off;
	const uint8_t* us = t.up_s[e] + off;
	const half2 gmg = __float2half2_rn( t.gate_g[e] * 16384.f );
	const half2 gmu = __float2half2_rn( t.up_g[e] * 16384.f );

	float cg[8], cu[8];
	#pragma unroll
	for ( int i = 0; i < 8; i++ ) { cg[i] = 0.f; cu[i] = 0.f; }

	#pragma unroll 5
	for ( int g = g0; g < g0 + Gq; g++ ) {
		const uint2 qg = __ldcs( gc + ( size_t ) g * 32 );
		const uint2 qu = __ldcs( uc + ( size_t ) g * 32 );
		const half2 scg = __hmul2( fp8e4m3_to_half2( __ldcs( gs + ( size_t ) g * 32 ) ), gmg );
		const half2 scu = __hmul2( fp8e4m3_to_half2( __ldcs( us + ( size_t ) g * 32 ) ), gmu );
		uint4 a01 = make_uint4( 0, 0, 0, 0 ), a23 = make_uint4( 0, 0, 0, 0 );
		if ( r == 0 ) {
			a01 = *reinterpret_cast<const uint4*>( sx + g * 16 );
			a23 = *reinterpret_cast<const uint4*>( sx + g * 16 + 8 );
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

	// Row 0 of the accumulator lives in lanes with (lane & 17) == 0, registers {0, 1, 4, 5}
	if ( ( lane & 17 ) == 0 ) {
		const int qp = ( lane >> 2 ) & 3;
		#pragma unroll
		for ( int j = 0; j < 4; j++ ) {
			const int i = ( j & 1 ) | ( ( j >> 1 ) << 2 );
			const int col = qp * 8 + ( ( i & 1 ) | ( ( ( lane >> 1 ) & 1 ) << 1 ) | ( ( i >> 2 ) << 2 ) );
			cs[warp][0][col] = cg[i];
			cs[warp][1][col] = cu[i];
		}
	}
	__syncthreads();
	if ( threadIdx.x < 32 ) {
		float g = 0.f, u = 0.f;
		#pragma unroll
		for ( int w = 0; w < WARPS; w++ ) { g += cs[w][0][threadIdx.x]; u += cs[w][1][threadIdx.x]; }
		const float gh = __half2float( __float2half( g ) ), uh = __half2float( __float2half( u ) );
		h[( size_t ) route * I + tile * 32 + threadIdx.x] = __float2half( gh / ( 1.0f + __expf( -gh ) ) * uh );
	}
}

// Down projection of one route and 32-column tile into partial[route][col]; the last block to finish a tile sums the routes in order, weighted
template <int WARPS>
__global__ __launch_bounds__( WARPS * 32 )
void nvfp4_moe_down( engine::modules::NVFP4ExpertTable t, const half* __restrict__ h, const int* __restrict__ route_idx, const half* __restrict__ route_w,
	int n_routes, float* __restrict__ partial, int* __restrict__ counters, half* __restrict__ out ) {
	const int route = blockIdx.y;
	const int e = route_idx[route];
	const int H = t.hidden, K = t.inter;
	const int tile = blockIdx.x;
	__shared__ float cs[WARPS][32];
	__shared__ int last;
	if ( e < t.num_local ) {
		extern __shared__ half sx[];
		for ( int i = threadIdx.x * 8; i < K; i += WARPS * 32 * 8 ) *reinterpret_cast<uint4*>( sx + i ) = *reinterpret_cast<const uint4*>( h + ( size_t ) route * K + i );
		__syncthreads();
		const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
		const int G = K >> 4, Gq = G / WARPS, g0 = warp * Gq;
		const int r = ( lane & 3 ) + ( ( lane & 16 ) ? 4 : 0 );
		const size_t off = ( size_t ) tile * G * 32 + lane;
		const uint2* dc = reinterpret_cast<const uint2*>( t.down_c[e] ) + off;
		const uint8_t* ds = t.down_s[e] + off;
		const half2 gm = __float2half2_rn( t.down_g[e] * 16384.f );
		float c[8];
		#pragma unroll
		for ( int i = 0; i < 8; i++ ) c[i] = 0.f;
		#pragma unroll
		for ( int g = g0; g < g0 + Gq; g++ ) {
			const uint2 q = __ldcs( dc + ( size_t ) g * 32 );
			const half2 sc = __hmul2( fp8e4m3_to_half2( __ldcs( ds + ( size_t ) g * 32 ) ), gm );
			uint4 a01 = make_uint4( 0, 0, 0, 0 ), a23 = make_uint4( 0, 0, 0, 0 );
			if ( r == 0 ) {
				a01 = *reinterpret_cast<const uint4*>( sx + g * 16 );
				a23 = *reinterpret_cast<const uint4*>( sx + g * 16 + 8 );
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
		if ( ( lane & 17 ) == 0 ) {
			const int qp = ( lane >> 2 ) & 3;
			#pragma unroll
			for ( int j = 0; j < 4; j++ ) {
				const int i = ( j & 1 ) | ( ( j >> 1 ) << 2 );
				const int col = ( i & 1 ) | ( ( ( lane >> 1 ) & 1 ) << 1 ) | ( ( i >> 2 ) << 2 );
				cs[warp][qp * 8 + col] = c[i];
			}
		}
		__syncthreads();
		if ( threadIdx.x < 32 ) {
			float v = 0.f;
			#pragma unroll
			for ( int w = 0; w < WARPS; w++ ) v += cs[w][threadIdx.x];
			partial[( size_t ) route * H + tile * 32 + threadIdx.x] = __half2float( __float2half( v ) );
		}
	}
	// Last block of the tile reduces over routes (fixed order)
	__threadfence();
	__syncthreads();
	if ( threadIdx.x == 0 ) last = atomicAdd( counters + tile, 1 ) == n_routes - 1;
	__syncthreads();
	if ( last && threadIdx.x < 32 ) {
		__threadfence();
		float acc = 0.f;
		for ( int k = 0; k < n_routes; k++ ) {
			if ( route_idx[k] < t.num_local ) acc += __half2float( route_w[k] ) * __ldcg( partial + ( size_t ) k * H + tile * 32 + threadIdx.x );
		}
		out[tile * 32 + threadIdx.x] = __float2half( acc );
		if ( threadIdx.x == 0 ) counters[tile] = 0;
	}
}

namespace engine::modules {

void NVFP4MoE::forward_row( const NVFP4ExpertTable& t, const half* x, const int* route_idx, const half* route_w, int n_routes,
	half* h, float* partial, int* counters, half* out, backend::stream_t stream ) {
	ENGINE_ASSERT( t.hidden % 64 == 0 && t.inter % 64 == 0 && t.inter % 32 == 0 && t.hidden % 32 == 0, "NVFP4 MoE: dims" );
	ENGINE_ASSERT( ( t.hidden / 16 ) % 16 == 0 && ( t.inter / 16 ) % 8 == 0, "NVFP4 MoE: group counts" );
	nvfp4_moe_gate_up<16><<<dim3( t.inter / 32, n_routes ), 512, t.hidden * sizeof( half ), stream>>>( t, x, route_idx, h );
	nvfp4_moe_down<8><<<dim3( t.hidden / 32, n_routes ), 256, t.inter * sizeof( half ), stream>>>( t, h, route_idx, route_w, n_routes, partial, counters, out );
}

} // namespace engine::modules
