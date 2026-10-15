#include "engine/modules/qsa.h"
#include "engine/modules/dsa.h"
#include "engine/backend/cuda/rope_math.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

constexpr int D = QSA::DIM;

__device__ __forceinline__ float block_sum128( float v ) {
	__shared__ float ws[4];
	#pragma unroll
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffff, v, o );
	__syncthreads();
	if ( ( threadIdx.x & 31 ) == 0 ) ws[threadIdx.x >> 5] = v;
	__syncthreads();
	return ws[0] + ws[1] + ws[2] + ws[3];
}

// Rotate-half rope on the first rot_dim dims of a 128-wide vector held one element per thread
__device__ __forceinline__ float rope_elem( float x, float partner_x, int d, int pos, int rot_dim, float theta ) {
	const int half_rot = rot_dim / 2;
	if ( d >= rot_dim ) return x;
	const int i = d % half_rot;
	float s, c;
	engine::math::rope_cos_sin<false>( pos, i, rot_dim, theta, c, s );
	return d < half_rot ? x * c - partner_x * s : x * c + partner_x * s;
}

// One block (128 threads) per (row, head): rmsnorm * (1 + w), rope; head HEADS is the raw key, stored as is
__global__ __launch_bounds__( 128 )
void qsa_prep_kernel( const half* __restrict__ qk, const half* __restrict__ qw, half* __restrict__ q_out, half* __restrict__ ring, int layer,
	const pod::BatchRows* __restrict__ rows, size_t ring_slot_stride, int rot_dim, float theta, float eps ) {
	const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
	const half* src = qk + static_cast<size_t>( r ) * QSA::PROJ + h * D;
	const int pos = rows->pos[r];
	if ( h == QSA::HEADS ) {
		half* dst = ring + static_cast<size_t>( rows->slot[r] ) * ring_slot_stride + ( static_cast<size_t>( layer ) * QSA::RING + ( pos % QSA::RING ) ) * D;
		dst[d] = src[d];
		return;
	}
	__shared__ float xs[D];
	const float x = __half2float( src[d] );
	xs[d] = x;
	const float ms = block_sum128( x * x ) / D;
	const float xn = __half2float( __float2half_rn( x * rsqrtf( ms + eps ) * ( 1.0f + __half2float( qw[d] ) ) ) );   // rounded to the key dtype before the rope
	xs[d] = xn;
	__syncthreads();
	const int half_rot = rot_dim / 2;
	const float partner = ( d < rot_dim ) ? xs[d < half_rot ? d + half_rot : d - half_rot] : 0.0f;
	q_out[( static_cast<size_t>( r ) * QSA::HEADS + h ) * D + d] = __float2half_rn( rope_elem( xn, partner, d, pos, rot_dim, theta ) );
}

// One block per row: only rows that complete a pool work
__global__ __launch_bounds__( 128 )
void qsa_pool_kernel( const half* __restrict__ ring, half* __restrict__ pool_plane, const half* __restrict__ kw, int layer, const pod::BatchRows* __restrict__ rows,
	size_t ring_slot_stride, size_t pool_slot_stride, int max_pools, int rot_dim, float theta, float eps ) {
	const int r = blockIdx.x, d = threadIdx.x;
	const int pos = rows->pos[r];
	if ( pos % QSA::POOL != QSA::POOL - 1 ) return;
	const int slot = rows->slot[r], pool = pos / QSA::POOL;
	const half* rk = ring + static_cast<size_t>( slot ) * ring_slot_stride + static_cast<size_t>( layer ) * QSA::RING * D;
	float acc = 0.0f;
	#pragma unroll
	for ( int j = 0; j < QSA::POOL; ++j ) acc += __half2float( rk[static_cast<size_t>( ( pool * QSA::POOL + j ) % QSA::RING ) * D + d] );
	// mean in fp32, rounded to the key dtype before the norm, as the reference does
	const float mean = __half2float( __float2half_rn( acc / QSA::POOL ) );
	__shared__ float xs[D];
	const float ms = block_sum128( mean * mean ) / D;
	const float xn = __half2float( __float2half_rn( mean * rsqrtf( ms + eps ) * ( 1.0f + __half2float( kw[d] ) ) ) );
	xs[d] = xn;
	__syncthreads();
	const int half_rot = rot_dim / 2;
	const float partner = ( d < rot_dim ) ? xs[d < half_rot ? d + half_rot : d - half_rot] : 0.0f;
	half* out = pool_plane + static_cast<size_t>( slot ) * pool_slot_stride + ( static_cast<size_t>( layer ) * max_pools + pool ) * D;
	out[d] = __float2half_rn( rope_elem( xn, partner, d, pool * QSA::POOL, rot_dim, theta ) );
}

// Grid (pool tiles of 128, rows): a thread scores one pool against the row's HEADS queries (held in shared memory)
__global__ __launch_bounds__( 128 )
void qsa_score_kernel( const half* __restrict__ q, const half* __restrict__ pool_plane, half* __restrict__ scores, int score_stride, int layer, int max_pools,
	const pod::BatchRows* __restrict__ rows, size_t pool_slot_stride, float scale ) {
	const int r = blockIdx.y;
	const int pos = rows->pos[r];
	const int bound = ( pos + 1 ) / QSA::POOL;
	if ( bound <= DSA::SEL_POOLS ) return;
	const int p = blockIdx.x * 128 + threadIdx.x;
	__shared__ float qs[QSA::HEADS * D];
	for ( int i = threadIdx.x; i < QSA::HEADS * D; i += 128 ) qs[i] = __half2float( q[static_cast<size_t>( r ) * QSA::HEADS * D + i] );
	__syncthreads();
	if ( p >= bound ) return;
	const half* key = pool_plane + static_cast<size_t>( rows->slot[r] ) * pool_slot_stride + ( static_cast<size_t>( layer ) * max_pools + p ) * D;
	float s[QSA::HEADS] = { 0.0f, 0.0f, 0.0f, 0.0f };
	#pragma unroll 4
	for ( int d = 0; d < D; d += 8 ) {
		const uint4 raw = *reinterpret_cast<const uint4*>( key + d );
		const half2* k2 = reinterpret_cast<const half2*>( &raw );
		float kf[8];
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) { const float2 f = __half22float2( k2[i] ); kf[2 * i] = f.x; kf[2 * i + 1] = f.y; }
		#pragma unroll
		for ( int h = 0; h < QSA::HEADS; ++h ) {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) s[h] = fmaf( qs[h * D + d + i], kf[i], s[h] );
		}
	}
	float total = 0.0f;
	#pragma unroll
	for ( int h = 0; h < QSA::HEADS; ++h ) total += fmaxf( s[h], 0.0f );
	scores[static_cast<size_t>( r ) * score_stride + p] = __float2half_rn( total * scale );
}

} // namespace

void QSA::prep( const backend::half_t* qk, const backend::half_t* q_norm_w, backend::half_t* q_out, backend::half_t* ring, int layer, int rows,
	const pod::BatchRows* d_rows, size_t ring_slot_stride, int rot_dim, float theta, float eps, backend::stream_t stream ) {
	qsa_prep_kernel<<<dim3( rows, HEADS + 1 ), 128, 0, stream>>>( reinterpret_cast<const half*>( qk ), reinterpret_cast<const half*>( q_norm_w ), reinterpret_cast<half*>( q_out ),
		reinterpret_cast<half*>( ring ), layer, d_rows, ring_slot_stride, rot_dim, theta, eps );
}

void QSA::pool_update( const backend::half_t* ring, backend::half_t* pool_plane, const backend::half_t* k_norm_w, int layer, int rows,
	const pod::BatchRows* d_rows, size_t ring_slot_stride, size_t pool_slot_stride, int max_pools, int rot_dim, float theta, float eps, backend::stream_t stream ) {
	qsa_pool_kernel<<<rows, 128, 0, stream>>>( reinterpret_cast<const half*>( ring ), reinterpret_cast<half*>( pool_plane ), reinterpret_cast<const half*>( k_norm_w ), layer, d_rows,
		ring_slot_stride, pool_slot_stride, max_pools, rot_dim, theta, eps );
}

void QSA::score( const backend::half_t* q, const backend::half_t* pool_plane, backend::half_t* scores, int score_stride, int layer, int max_pools, int rows,
	const pod::BatchRows* d_rows, size_t pool_slot_stride, backend::stream_t stream ) {
	qsa_score_kernel<<<dim3( ( max_pools + 127 ) / 128, rows ), 128, 0, stream>>>( reinterpret_cast<const half*>( q ), reinterpret_cast<const half*>( pool_plane ), reinterpret_cast<half*>( scores ),
		score_stride, layer, max_pools, d_rows, pool_slot_stride, 1.0f / sqrtf( static_cast<float>( D ) ) );
}

} // namespace engine::modules
