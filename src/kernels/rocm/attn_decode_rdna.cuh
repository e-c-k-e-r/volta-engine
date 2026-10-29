#pragma once
// ROCm only (RDNA, wave32). Included by src/backend/cuda/modules/attention.cu inside its anonymous namespace, after the KV helpers and
// ATTN_* constants it uses.
// Decode attention, one row per block column (rows of distinct slots), head_dim 256: a block owns (kv head, row, split) and all G <= 8 query
// heads of the kv head, so every key and value is read once for the G heads. Its 4 waves take tiles of AD_TK = 8 keys of the split's range
// in turn, each tile two rounds of loads issued together:
//   scores   lanes 4j .. 4j + 3 take key j of the tile, a quarter of its K row each, and dot it with the G queries (v_dot2 against the
//            queries in shared memory); a quad sum completes the score
//   softmax  per head the tile maximum is a wave reduction; the running sums stay per lane until the end
//   values   fetched while the softmax runs; the probabilities go through shared memory, lane l accumulates output dimensions 8l .. 8l + 7
//            of the G heads
// The waves' states are merged in shared memory and the block writes the split's unnormalized partial (m in natural-log units, l, O) in
// the layout attn_reduce_kernel reads, which adds the gate. The key range of a split is computed from the row's position, so the grid only
// depends on the row count (graph replay).
#define AD_W 4
#define AD_GMAX 8
#define AD_TK 8   // keys per wave tile

template <KVQ KQ, KVQ VQ>
__global__ __launch_bounds__( 32 * AD_W )
void attn_decode_rdna_kernel(
	const half* __restrict__ q,
	const pod::KVRef kv,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* __restrict__ rows,
	int num_q_heads,
	int num_kv_heads,
	int max_seq,
	float scale,
	float* __restrict__ workspace,
	int nsplit
) {
	constexpr int HD = 256;
	__shared__ __align__( 16 ) half sQ[AD_GMAX][HD];
	__shared__ float sP[AD_W][AD_TK][AD_GMAX];
	__shared__ float sM[AD_W][AD_GMAX], sL[AD_W][AD_GMAX];
	__shared__ float sO[AD_W][AD_GMAX][HD];

	const int G = num_q_heads / num_kv_heads;
	const int kvh = blockIdx.x, row = blockIdx.y, split = blockIdx.z;
	const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
	const int pos = min( rows->pos[row], max_seq - 1 );
	const int total = pos + 1;
	const int len = ( ( total + nsplit - 1 ) / nsplit + AD_TK - 1 ) / AD_TK * AD_TK;
	const int k0 = min( total, split * len ), k1 = min( total, k0 + len );

	const size_t kv_stride = static_cast<size_t>( num_kv_heads ) * HD;
	const size_t kv_base = static_cast<size_t>( rows->slot[row] ) * slot_stride + static_cast<size_t>( layer_idx ) * max_seq * kv_stride +
		static_cast<size_t>( kvh ) * HD;

	for ( int u = tid; u < G * ( HD / 8 ); u += 32 * AD_W ) {
		const int h = u / ( HD / 8 ), c = u % ( HD / 8 );
		*reinterpret_cast<uint4*>( &sQ[h][c * 8] ) =
			*reinterpret_cast<const uint4*>( q + ( static_cast<size_t>( row ) * num_q_heads + kvh * G + h ) * HD + c * 8 );
	}
	__syncthreads();

	const float scale2 = scale * 1.4426950408889634f;
	float m_run[AD_GMAX], l_lane[AD_GMAX], o[AD_GMAX][8];
	#pragma unroll
	for ( int h = 0; h < AD_GMAX; ++h ) {
		m_run[h] = -1e30f;
		l_lane[h] = 0.0f;
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) o[h][i] = 0.0f;
	}

	for ( int tb = k0 + wave * AD_TK; tb < k1; tb += AD_TK * AD_W ) {
		// Scores: key tb + (lane >> 2), lane & 3 its quarter of the dimensions (64); the 8 loads of a lane are issued together
		const int kk = lane >> 2, part = lane & 3;
		const int t = tb + kk;
		const bool valid = t < k1;
		const size_t krow = kv_base + static_cast<size_t>( valid ? t : k0 ) * kv_stride + part * 64;
		KVRaw8 rk[8];
		#pragma unroll
		for ( int c = 0; c < 8; ++c ) rk[c] = kv_fetch8<KQ>( kv.k, kv.ks, krow + c * 8 );
		float acc[AD_GMAX];
		#pragma unroll
		for ( int h = 0; h < AD_GMAX; ++h ) acc[h] = 0.0f;
		#pragma unroll
		for ( int c = 0; c < 8; ++c ) {
			const uint4 kh = kv_decode8_h<KQ>( rk[c] );
			const half2* k2 = reinterpret_cast<const half2*>( &kh );
			#pragma unroll
			for ( int h = 0; h < AD_GMAX; ++h ) {
				if ( h >= G ) break;
				const uint4 qh = *reinterpret_cast<const uint4*>( &sQ[h][part * 64 + c * 8] );
				const half2* q2 = reinterpret_cast<const half2*>( &qh );
				#pragma unroll
				for ( int i = 0; i < 4; ++i ) acc[h] = __builtin_amdgcn_fdot2( __builtin_bit_cast( wm_v2h, q2[i] ), __builtin_bit_cast( wm_v2h, k2[i] ), acc[h], false );
			}
		}
		// The tile's values are fetched while the softmax runs: lane holds dimensions 8 lane .. 8 lane + 7
		const int nk = min( AD_TK, k1 - tb );
		KVRaw8 rv[AD_TK];
		#pragma unroll
		for ( int k = 0; k < AD_TK; ++k ) rv[k] = kv_fetch8<VQ>( kv.v, kv.vs, kv_base + static_cast<size_t>( tb + min( k, nk - 1 ) ) * kv_stride + lane * 8 );
		// Online softmax over the tile (the 4 lanes of a key hold the same score after the quad sum)
		#pragma unroll
		for ( int h = 0; h < AD_GMAX; ++h ) {
			if ( h >= G ) break;
			float a = acc[h];
			a += __shfl_xor( a, 1 );
			a += __shfl_xor( a, 2 );
			const float s = valid ? a * scale2 : -1e30f;
			float mx = s;
			#pragma unroll
			for ( int d = 4; d <= 16; d <<= 1 ) mx = fmaxf( mx, __shfl_xor( mx, d ) );
			const float m_new = fmaxf( m_run[h], mx );
			const float alpha = exp2f( m_run[h] - m_new );
			const float p = valid ? exp2f( s - m_new ) : 0.0f;
			l_lane[h] = l_lane[h] * alpha + ( part == 0 ? p : 0.0f );
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) o[h][i] *= alpha;
			m_run[h] = m_new;
			if ( part == 0 ) sP[wave][kk][h] = p;
		}
		// the wave writes and reads its own probabilities: a wave's LDS accesses complete in order, only the compiler must keep them apart
		asm volatile( "" ::: "memory" );
		#pragma unroll
		for ( int k = 0; k < AD_TK; ++k ) {
			if ( k >= nk ) break;
			const uint4 vh = kv_decode8_h<VQ>( rv[k] );
			const half2* v2 = reinterpret_cast<const half2*>( &vh );
			float v[8];
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const float2 f = __half22float2( v2[i] );
				v[2 * i] = f.x;
				v[2 * i + 1] = f.y;
			}
			#pragma unroll
			for ( int h = 0; h < AD_GMAX; ++h ) {
				if ( h >= G ) break;
				const float p = sP[wave][k][h];
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) o[h][i] = fmaf( p, v[i], o[h][i] );
			}
		}
		asm volatile( "" ::: "memory" );
	}

	// Merge the waves: m is wave-uniform, l is summed over the lanes
	#pragma unroll
	for ( int h = 0; h < AD_GMAX; ++h ) {
		if ( h >= G ) break;
		float l = l_lane[h];
		#pragma unroll
		for ( int d = 16; d >= 1; d >>= 1 ) l += __shfl_xor( l, d );
		if ( lane == 0 ) {
			sM[wave][h] = m_run[h];
			sL[wave][h] = l;
		}
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) sO[wave][h][lane * 8 + i] = o[h][i];
	}
	__syncthreads();
	for ( int e = tid; e < G * HD; e += 32 * AD_W ) {
		const int h = e / HD, d = e % HD;
		float m = -1e30f;
		#pragma unroll
		for ( int w = 0; w < AD_W; ++w ) m = fmaxf( m, sM[w][h] );
		float l = 0.0f, ov = 0.0f;
		#pragma unroll
		for ( int w = 0; w < AD_W; ++w ) {
			const float f = sL[w][h] > 0.0f ? exp2f( sM[w][h] - m ) : 0.0f;
			l += f * sL[w][h];
			ov += f * sO[w][h][d];
		}
		float* ws = workspace + ( ( static_cast<size_t>( row ) * num_q_heads + kvh * G + h ) * nsplit + split ) * ( ATTN_MAX_HEAD_DIM + 2 );
		if ( d == 0 ) {
			ws[0] = l > 0.0f ? m * 0.6931471805599453f : -1e30f;
			ws[1] = l;
		}
		ws[2 + d] = ov;
	}
}
