// Training forms of GLM-5.3-Flash operations: the MoE routing (sigmoid scores, biased top-k selection, normalized weights) and the routed
// experts' SwiGLU with the clamp after the SiLU, each forward with its backward (engine/train/glm5_moe.h)
#include "engine/train/glm5_moe.h"
#include "engine/config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace engine::train::glm5_next {

namespace {

#define G5M_LOOP( i, n ) for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ( n ); i += (size_t) gridDim.x * blockDim.x )

constexpr int kMaxK = 16;
constexpr float kRouteEps = 1e-9f;

unsigned blocks_for( size_t n ) { return static_cast<unsigned>( ( n + 255 ) / 256 < 65535u * 8u ? ( n + 255 ) / 256 : 65535u * 8u ); }

cudaStream_t cu( backend::stream_t s ) { return static_cast<cudaStream_t>( s ); }

__host__ __device__ __forceinline__ const float4* f4( const float* p ) { return reinterpret_cast<const float4*>( p ); }
__host__ __device__ __forceinline__ float4* f4( float* p ) { return reinterpret_cast<float4*>( p ); }

__device__ __forceinline__ float sigmoidf_( float x ) { return 1.0f / ( 1.0f + expf( -x ) ); }
__device__ __forceinline__ float4 mul4( float4 a, float4 b ) { return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); }

// ---- routing ----
// One block per row, a thread per expert: k rounds of a block argmax of s + bias (ties to the lowest index), the selected s kept for the
// weights, summed in selection order as the inference router does
__global__ void k_glm_route( const float* logits, const float* bias, int experts, int k, float scale, int* ids, float* w ) {
	__shared__ float s_v[32];
	__shared__ int s_i[32];
	__shared__ float s_s[kMaxK];
	const int r = blockIdx.x, e = threadIdx.x;
	const float s = e < experts ? sigmoidf_( logits[static_cast<size_t>( r ) * experts + e] ) : 0.0f;
	const float key = e < experts ? s + ( bias ? bias[e] : 0.0f ) : -INFINITY;
	bool taken = e >= experts;
	for ( int j = 0; j < k; ++j ) {
		float v = taken ? -INFINITY : key;
		int idx = taken ? 0x7fffffff : e;
		for ( int o = 16; o > 0; o >>= 1 ) {
			const float ov = __shfl_xor_sync( 0xffffffffu, v, o );
			const int oi = __shfl_xor_sync( 0xffffffffu, idx, o );
			if ( ov > v || ( ov == v && oi < idx ) ) { v = ov; idx = oi; }
		}
		if ( ( e & 31 ) == 0 ) { s_v[e >> 5] = v; s_i[e >> 5] = idx; }
		__syncthreads();
		if ( e == 0 ) {
			float bv = s_v[0];
			int bi = s_i[0];
			for ( int i = 1; i < ( blockDim.x >> 5 ); ++i ) if ( s_v[i] > bv || ( s_v[i] == bv && s_i[i] < bi ) ) { bv = s_v[i]; bi = s_i[i]; }
			ids[static_cast<size_t>( r ) * k + j] = bi;
			s_i[0] = bi;
		}
		__syncthreads();
		if ( e == s_i[0] ) { taken = true; s_s[j] = s; }
		__syncthreads();
	}
	if ( e == 0 ) {
		float total = 0.0f;
		for ( int j = 0; j < k; ++j ) total += s_s[j];
		const float inv = scale / ( total + kRouteEps );
		for ( int j = 0; j < k; ++j ) w[static_cast<size_t>( r ) * k + j] = s_s[j] * inv;
	}
}

// A thread per row: the weights at the given selection (the forward's arithmetic)
__global__ void k_glm_route_given( const float* logits, const int* ids, int rows, int experts, int k, float scale, float* w ) {
	G5M_LOOP( r, (size_t) rows ) {
		float s[kMaxK], total = 0.0f;
		for ( int j = 0; j < k; ++j ) {
			s[j] = sigmoidf_( logits[r * experts + ids[r * k + j]] );
			total += s[j];
		}
		const float inv = scale / ( total + kRouteEps );
		for ( int j = 0; j < k; ++j ) w[r * k + j] = s[j] * inv;
	}
}

// A thread per row: dL/ds_j = ( scale dw_j - sum_j' w_j' dw_j' ) / S, times sigmoid'( logit_j ) = s_j sigmoid( -logit_j )
__global__ void k_glm_route_bwd( const float* logits, const int* ids, int rows, int experts, int k, float scale, const float* dw, float* dlogit ) {
	G5M_LOOP( r, (size_t) rows ) {
		float x[kMaxK], s[kMaxK], total = 0.0f;
		for ( int j = 0; j < k; ++j ) {
			x[j] = logits[r * experts + ids[r * k + j]];
			s[j] = sigmoidf_( x[j] );
			total += s[j];
		}
		const float S = total + kRouteEps, inv = scale / S;
		float t = 0.0f;
		for ( int j = 0; j < k; ++j ) t += s[j] * inv * dw[r * k + j];
		for ( int j = 0; j < k; ++j ) dlogit[r * k + j] = s[j] * sigmoidf_( -x[j] ) * ( scale * dw[r * k + j] - t ) / S;
	}
}

// ---- SwiGLU, clamp after the SiLU ----
__device__ __forceinline__ float act_after( float g, float u, float limit ) {
	float a = g * sigmoidf_( g );
	if ( limit > 0.0f ) {
		a = fminf( a, limit );
		u = fminf( fmaxf( u, -limit ), limit );
	}
	return a * u;
}

__device__ __forceinline__ void act_after_grad( float g, float u, float da, float limit, float& dg, float& du ) {
	const float s = sigmoidf_( g );
	float a = g * s, uc = u;
	bool pass_g = true, pass_u = true;
	if ( limit > 0.0f ) {
		if ( a >= limit ) a = limit, pass_g = false;
		if ( u >= limit ) uc = limit, pass_u = false;
		if ( u <= -limit ) uc = -limit, pass_u = false;
	}
	dg = pass_g ? da * uc * s * ( 1.0f + g * ( 1.0f - s ) ) : 0.0f;
	du = pass_u ? da * a : 0.0f;
}

__global__ void k_swiglu_after( const float* g, const float* u, float* a, size_t n, float limit ) {
	G5M_LOOP( i, n ) a[i] = act_after( g[i], u[i], limit );
}

__global__ void k_swiglu_after_bwd( const float* g, const float* u, const float* da, float* dg, float* du, size_t n, float limit ) {
	G5M_LOOP( i, n ) act_after_grad( g[i], u[i], da[i], limit, dg[i], du[i] );
}

// Grouped routes, rows in float4s (cols4 = cols / 4)
__global__ void k_route_swiglu_after( const float4* Zg, const float4* Zu, const int* gexp, int m, const float* const* svg, const float* const* svu, int P,
	int cols4, float limit, float4* G, float4* U, float4* A ) {
	G5M_LOOP( i, (size_t) P * cols4 ) {
		const int e = gexp[i / cols4 / m];
		const size_t c = i % cols4;
		const float4 g = mul4( Zg[i], f4( svg[e] )[c] ), u = mul4( Zu[i], f4( svu[e] )[c] );
		G[i] = g;
		U[i] = u;
		A[i] = make_float4( act_after( g.x, u.x, limit ), act_after( g.y, u.y, limit ), act_after( g.z, u.z, limit ), act_after( g.w, u.w, limit ) );
	}
}

__global__ void k_route_swiglu_after_bwd( const float4* dAk, const float4* G, const float4* U, const int* gexp, int m, const float* const* sud, int P,
	int cols4, float limit, float4* dG, float4* dU ) {
	G5M_LOOP( i, (size_t) P * cols4 ) {
		const int e = gexp[i / cols4 / m];
		const size_t c = i % cols4;
		const float4 da = mul4( dAk[i], f4( sud[e] )[c] ), g = G[i], u = U[i];
		float4 dg, du;
		act_after_grad( g.x, u.x, da.x, limit, dg.x, du.x );
		act_after_grad( g.y, u.y, da.y, limit, dg.y, du.y );
		act_after_grad( g.z, u.z, da.z, limit, dg.z, du.z );
		act_after_grad( g.w, u.w, da.w, limit, dg.w, du.w );
		dG[i] = dg;
		dU[i] = du;
	}
}

} // namespace

void glm_route_forward( const float* logits, const float* bias, int rows, int experts, int k, float scale, int* ids, float* w, backend::stream_t stream ) {
	ENGINE_ASSERT( experts <= 1024 && k <= kMaxK && k <= experts, "glm_route_forward: up to 1024 experts and 16 routes" );
	if ( rows > 0 ) k_glm_route<<<rows, ( experts + 31 ) / 32 * 32, 0, cu( stream )>>>( logits, bias, experts, k, scale, ids, w );
}

void glm_route_given( const float* logits, const int* ids, int rows, int experts, int k, float scale, float* w, backend::stream_t stream ) {
	ENGINE_ASSERT( k <= kMaxK, "glm_route_given: up to 16 routes" );
	if ( rows > 0 ) k_glm_route_given<<<blocks_for( rows ), 256, 0, cu( stream )>>>( logits, ids, rows, experts, k, scale, w );
}

void glm_route_backward( const float* logits, const int* ids, int rows, int experts, int k, float scale, const float* dw, float* dlogit,
	backend::stream_t stream ) {
	ENGINE_ASSERT( k <= kMaxK, "glm_route_backward: up to 16 routes" );
	if ( rows > 0 ) k_glm_route_bwd<<<blocks_for( rows ), 256, 0, cu( stream )>>>( logits, ids, rows, experts, k, scale, dw, dlogit );
}

void swiglu_after_forward( const float* g, const float* u, float* a, size_t n, float limit, backend::stream_t stream ) {
	if ( n > 0 ) k_swiglu_after<<<blocks_for( n ), 256, 0, cu( stream )>>>( g, u, a, n, limit );
}

void swiglu_after_backward( const float* g, const float* u, const float* da, float* dg, float* du, size_t n, float limit, backend::stream_t stream ) {
	if ( n > 0 ) k_swiglu_after_bwd<<<blocks_for( n ), 256, 0, cu( stream )>>>( g, u, da, dg, du, n, limit );
}

void glm_route_swiglu( const float* Zg, const float* Zu, const int* gexp, int m, const float* const* svg, const float* const* svu, int P, int cols, float limit,
	float* G, float* U, float* A, backend::stream_t stream ) {
	ENGINE_ASSERT( cols % 4 == 0, "glm_route_swiglu: route rows must be float4 multiples" );
	if ( P > 0 )
		k_route_swiglu_after<<<blocks_for( (size_t) P * cols / 4 ), 256, 0, cu( stream )>>>( f4( Zg ), f4( Zu ), gexp, m, svg, svu, P, cols / 4, limit, f4( G ),
			f4( U ), f4( A ) );
}

void glm_route_swiglu_backward( const float* dAk, const float* G, const float* U, const int* gexp, int m, const float* const* sud, int P, int cols, float limit,
	float* dG, float* dU, backend::stream_t stream ) {
	ENGINE_ASSERT( cols % 4 == 0, "glm_route_swiglu_backward: route rows must be float4 multiples" );
	if ( P > 0 )
		k_route_swiglu_after_bwd<<<blocks_for( (size_t) P * cols / 4 ), 256, 0, cu( stream )>>>( f4( dAk ), f4( G ), f4( U ), gexp, m, sud, P, cols / 4, limit,
			f4( dG ), f4( dU ) );
}

namespace {

__global__ void k_upload_words( const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, size_t n ) {
	for ( size_t i = blockIdx.x * static_cast<size_t>( blockDim.x ) + threadIdx.x; i < n; i += static_cast<size_t>( gridDim.x ) * blockDim.x ) dst[i] = src[i];
}

} // namespace

void upload_words( const void* mapped, void* dst, size_t bytes, backend::stream_t stream ) {
	ENGINE_ASSERT( bytes % 4 == 0, "upload_words: whole 4-byte words" );
	const size_t n = bytes / 4;
	if ( n > 0 )
		k_upload_words<<<static_cast<unsigned>( std::min<size_t>( ( n + 255 ) / 256, 1024 ) ), 256, 0, cu( stream )>>>( static_cast<const uint32_t*>( mapped ),
			static_cast<uint32_t*>( dst ), n );
}

} // namespace engine::train::glm5_next
