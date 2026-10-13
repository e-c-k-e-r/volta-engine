#pragma once
// ROCm only. Included by src/backend/cuda/modules/norm.cu inside its anonymous namespace.
// 16-byte loads for rows of up to 8192 elements (a multiple of 8): the scalar kernel above issues 20 two-byte loads and 20 weight loads
// per thread at the Qwen width, 10 us a call on RDNA3 where this takes a few. The sum is accumulated in a different order, so the
// result differs from rms_norm_kernel in the last bit of the scale
__global__ __launch_bounds__( 256 ) void rms_norm_vec_kernel(
	const half* __restrict__ src,
	half* __restrict__ dst,
	const half* __restrict__ weight,
	int hidden_size,
	float eps,
	float constant_bias
) {
	constexpr int KM = 4;
	const int row = blockIdx.x;
	const int tid = threadIdx.x;
	const int nvec = hidden_size >> 3;
	const uint4* rs = reinterpret_cast<const uint4*>( src + static_cast<int64_t>( row ) * hidden_size );
	uint4* rd = reinterpret_cast<uint4*>( dst + static_cast<int64_t>( row ) * hidden_size );
	const uint4* wv = reinterpret_cast<const uint4*>( weight );
	uint4 raw[KM], wraw[KM];
	float local_sq = 0.0f;
	#pragma unroll
	for ( int k = 0; k < KM; ++k ) {
		const int i = tid + k * 256;
		raw[k] = make_uint4( 0, 0, 0, 0 );
		wraw[k] = make_uint4( 0, 0, 0, 0 );
		if ( i < nvec ) {
			raw[k] = rs[i];
			if ( weight != nullptr ) wraw[k] = wv[i];
		}
	}
	#pragma unroll
	for ( int k = 0; k < KM; ++k ) {
		const half2* h2 = reinterpret_cast<const half2*>( &raw[k] );
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const float2 f = __half22float2( h2[j] );
			local_sq = fmaf( f.x, f.x, fmaf( f.y, f.y, local_sq ) );
		}
	}
	#pragma unroll
	for ( int mask = 16; mask > 0; mask >>= 1 ) local_sq += __shfl_xor_sync( 0xffffffff, local_sq, mask );
	__shared__ float s_warp_sums[8];
	const int warp_id = tid >> 5, lane_id = tid & 31;
	if ( lane_id == 0 ) s_warp_sums[warp_id] = local_sq;
	__syncthreads();
	float b_sum = ( lane_id < 8 ) ? s_warp_sums[lane_id] : 0.0f;
	#pragma unroll
	for ( int mask = 4; mask > 0; mask >>= 1 ) b_sum += __shfl_xor_sync( 0xffffffff, b_sum, mask );
	const float inv_rms = rsqrtf( __shfl_sync( 0xffffffff, b_sum, 0 ) / static_cast<float>( hidden_size ) + eps );
	#pragma unroll
	for ( int k = 0; k < KM; ++k ) {
		const int i = tid + k * 256;
		if ( i >= nvec ) continue;
		const half2* h2 = reinterpret_cast<const half2*>( &raw[k] );
		const half2* w2 = reinterpret_cast<const half2*>( &wraw[k] );
		uint4 out;
		half2* o2 = reinterpret_cast<half2*>( &out );
		#pragma unroll
		for ( int j = 0; j < 4; ++j ) {
			const float2 f = __half22float2( h2[j] );
			float2 w = make_float2( 1.0f, 1.0f );
			if ( weight != nullptr ) {
				const float2 wf = __half22float2( w2[j] );
				w = make_float2( wf.x + constant_bias, wf.y + constant_bias );
			}
			o2[j] = __floats2half2_rn( f.x * w.x * inv_rms, f.y * w.y * inv_rms );
		}
		rd[i] = out;
	}
}
