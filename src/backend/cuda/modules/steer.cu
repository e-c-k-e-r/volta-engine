#include "engine/modules/steer.h"
#include "engine/backend/cuda/reduce.cuh"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

__device__ __forceinline__ int row_class( const SteerSlotCtl& c, int pos ) {
	return pos < c.prompt_len ? STEER_PROMPT : ( pos < c.think_end ? STEER_THINK : STEER_ANSWER );
}

// One block per row; the row is edited in shared memory (fp32) through the slot's entries in order, then written back
__global__ __launch_bounds__( 256 )
void steer_apply_kernel( SteerTables t, half* __restrict__ hidden, const pod::BatchRows* __restrict__ rows, int layer_index ) {
	extern __shared__ float srow[];
	const int r = blockIdx.x;
	const int slot = rows->slot[r];
	const SteerSlotCtl c = t.ctl[slot];
	if ( ( c.scope & row_class( c, rows->pos[r] ) ) == 0 ) return;
	const size_t entry0 = ( static_cast<size_t>( slot ) * t.layers + layer_index ) * kSteerEntries;
	bool any = false;
	#pragma unroll
	for ( int k = 0; k < kSteerEntries; ++k ) any |= t.kind[entry0 + k] != STEER_OFF;
	if ( !any ) return;

	half* x = hidden + static_cast<size_t>( r ) * t.hidden;
	for ( int i = threadIdx.x; i < t.hidden; i += blockDim.x ) srow[i] = __half2float( x[i] );
	__syncthreads();
	for ( int k = 0; k < kSteerEntries; ++k ) {
		const int kind = t.kind[entry0 + k];
		if ( kind == STEER_OFF ) continue;
		const float* v = t.vec + ( entry0 + k ) * t.hidden;
		const float par = t.par[entry0 + k];
		if ( kind == STEER_ADD ) {
			for ( int i = threadIdx.x; i < t.hidden; i += blockDim.x ) srow[i] += par * v[i];
		}
		else {
			float dot = 0.0f;
			for ( int i = threadIdx.x; i < t.hidden; i += blockDim.x ) dot = fmaf( srow[i], v[i], dot );
			dot = engine::math::block_sum<8>( dot );
			const float over = dot - par;
			if ( over > 0.0f ) {
				for ( int i = threadIdx.x; i < t.hidden; i += blockDim.x ) srow[i] -= over * v[i];
			}
			__syncthreads();   // the next block_sum reuses its shared array, and the next entry reads the edited row
		}
		__syncthreads();
	}
	for ( int i = threadIdx.x; i < t.hidden; i += blockDim.x ) x[i] = __float2half_rn( fminf( fmaxf( srow[i], -65504.0f ), 65504.0f ) );
}

__global__ __launch_bounds__( 128 )
void steer_penalize_kernel( SteerTables t, half* __restrict__ logits, int logit_cols, const pod::BatchRows* __restrict__ rows, int row0,
	int shard_lo, int shard_cols ) {
	const int j = blockIdx.x;
	const int slot = rows->slot[row0 + j];
	const SteerSlotCtl c = t.ctl[slot];
	if ( c.penalty_n <= 0 || c.penalty == 0.0f ) return;
	const int predicted = rows->pos[row0 + j] + 1;   // the token this row's logits choose
	if ( predicted < c.prompt_len || predicted >= c.think_end ) return;
	const int32_t* ids = t.penalty_ids + static_cast<size_t>( slot ) * kSteerPenaltyIds;
	half* row = logits + static_cast<size_t>( j ) * logit_cols;
	for ( int i = threadIdx.x; i < c.penalty_n; i += blockDim.x ) {
		const int local = ids[i] - shard_lo;
		if ( local >= 0 && local < shard_cols ) row[local] = __float2half_rn( __half2float( row[local] ) - c.penalty );
	}
}

}   // namespace

void ResidualSteer::apply( const SteerTables& tables, backend::half_t* hidden, const pod::BatchRows* d_rows, int rows, int layer_index, backend::stream_t stream ) {
	steer_apply_kernel<<<rows, 256, static_cast<size_t>( tables.hidden ) * sizeof( float ), stream>>>( tables, reinterpret_cast<half*>( hidden ), d_rows, layer_index );
}

void ResidualSteer::penalize( const SteerTables& tables, backend::half_t* logits, int logit_cols, const pod::BatchRows* d_rows, int row0, int count,
	int shard_lo, int shard_cols, backend::stream_t stream ) {
	steer_penalize_kernel<<<count, 128, 0, stream>>>( tables, reinterpret_cast<half*>( logits ), logit_cols, d_rows, row0, shard_lo, shard_cols );
}

}   // namespace engine::modules
