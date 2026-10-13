#pragma once
// ROCm only (RDNA3, wave32). Included by src/backend/cuda/modules/attention.cu inside its anonymous namespace, after the KV helpers,
// ATTN_* constants and attn_reduce_kernel it uses.
// RDNA3 prefill / verify / decode attention on the native wave32 WMMA (v_wmma_f32_16x16x16_f16). Per kv head a block owns WM_BQ
// query slots (row, head pairs; 16 per wave) and walks the keys in tiles of WM_BK; the work is done transposed so the softmax is
// per lane: S^T = K Q^T puts one query in each lane's column (rows = keys 2i + half), so the row max / sum are in-register plus one
// exchange with the lane 16 away, and P^T is already the B operand of O^T = V^T P^T (the two halves' keys interleave back to the
// natural order). V is staged transposed (dimension-major) so an A fragment is one contiguous 32 bytes. head_dim 256, 64 KB of
// shared memory. Same partial layout as attn_chunked_paged_kernel when nsplit > 1.
#define WM_W 3
#define WM_BQ ( 16 * WM_W )
#define WM_BK 16
#define WM_KS 264
#define WM_VS ( WM_BK + 8 )
#define WM_OS 257
#define WM_OFF_K ( WM_BQ * WM_KS * 2 )
#define WM_OFF_VT ( WM_OFF_K + WM_BK * WM_KS * 2 )
#define WM_SMEM_TILES ( WM_OFF_VT + 256 * WM_VS * 2 )
#define WM_SMEM ( WM_SMEM_TILES > WM_BQ * WM_OS * 4 ? WM_SMEM_TILES : WM_BQ * WM_OS * 4 )   // the output staging reuses the tiles
static_assert( WM_SMEM <= 65536, "attn_wmma_kernel: shared memory" );

template <KVQ KQ, KVQ VQ>
__global__ __launch_bounds__( 32 * WM_W )
void attn_wmma_kernel(
	const half* __restrict__ q,
	const pod::KVRef kv,
	const half* __restrict__ gate,
	half* __restrict__ out,
	size_t slot_stride,
	int layer_idx,
	const pod::BatchRows* __restrict__ rows,
	int count,
	int num_q_heads,
	int num_kv_heads,
	int max_seq,
	float scale,
	float* __restrict__ workspace,
	int nsplit,
	int tile_q   // query slots a block owns: WM_BQ, or G to give every row its own block (rows of distinct slots)
) {
	constexpr int HD = 256;
	constexpr int T = 32 * WM_W;
	extern __shared__ __align__( 16 ) unsigned char wm_smem[];
	half* sQ = reinterpret_cast<half*>( wm_smem );
	half* sK = reinterpret_cast<half*>( wm_smem + WM_OFF_K );
	half* sVt = reinterpret_cast<half*>( wm_smem + WM_OFF_VT );
	float* sO = reinterpret_cast<float*>( wm_smem );

	const int G = num_q_heads / num_kv_heads;
	const int kvh = blockIdx.x;
	const int q_base = blockIdx.y * tile_q;
	const int q_end = min( q_base + tile_q, count * G );
	const int tid = threadIdx.x;
	const int wave = tid >> 5;
	const int lane = tid & 31;
	const int ql = lane & 15;
	const int hi = lane >> 4;

	const int last_row = ( q_end - 1 ) / G;
	int max_pos = 0;
	for ( int r = q_base / G; r <= last_row; ++r ) max_pos = max( max_pos, rows->pos[r] );
	max_pos = min( max_pos, max_seq - 1 );
	const size_t slot_off = static_cast<size_t>( rows->slot[q_base / G] ) * slot_stride;
	const size_t kv_stride = static_cast<size_t>( num_kv_heads ) * HD;
	const size_t kv_base = slot_off + static_cast<size_t>( layer_idx ) * max_seq * kv_stride + static_cast<size_t>( kvh ) * HD;

	const int qs = q_base + wave * 16 + ql;   // this lane's query slot (its WMMA column)
	const bool wave_active = q_base + wave * 16 < q_end;
	const int spos = qs < q_end ? min( rows->pos[qs / G], max_seq - 1 ) : -1;

	for ( int u = tid; u < WM_BQ * ( HD / 8 ); u += T ) {
		const int qq = u >> 5, c = u & 31;
		const int qi = q_base + qq;
		uint4 raw = make_uint4( 0, 0, 0, 0 );
		if ( qi < q_end ) raw = *reinterpret_cast<const uint4*>( q + ( static_cast<size_t>( qi / G ) * num_q_heads + kvh * G + qi % G ) * HD + c * 8 );
		*reinterpret_cast<uint4*>( sQ + qq * WM_KS + c * 8 ) = raw;
	}

	wm_v8f o[16];
	#pragma unroll
	for ( int d = 0; d < 16; ++d ) o[d] = wm_v8f{ 0, 0, 0, 0, 0, 0, 0, 0 };
	float m_run = -1e30f, l_run = 0.0f;
	const float scale2 = scale * 1.4426950408889634f;

	const int ntiles = ( max_pos + WM_BK ) / WM_BK;
	const int kb0 = ( blockIdx.z * ntiles / nsplit ) * WM_BK;
	const int kb1 = ( ( blockIdx.z + 1 ) * ntiles / nsplit ) * WM_BK;

	// The next tile's keys and values are fetched into registers while this one is computed on, then decoded into shared memory
	constexpr int NKT = WM_BK / 16;
	constexpr int KI = ( WM_BK * 32 + T - 1 ) / T;
	constexpr int VI = ( ( WM_BK / 2 ) * 32 + T - 1 ) / T;
	KVRaw8 rk[KI], rv[VI][2];
	auto fetch = [&]( int kb ) {
		#pragma unroll
		for ( int i = 0; i < KI; ++i ) {
			const int u = tid + i * T, key = u >> 5, c = u & 31;
			rk[i].w = make_uint4( 0, 0, 0, 0 );
			rk[i].s = 0;
			if ( u < WM_BK * 32 && kb + key <= max_pos ) rk[i] = kv_fetch8<KQ>( kv.k, kv.ks, kv_base + static_cast<size_t>( kb + key ) * kv_stride + c * 8 );
		}
		#pragma unroll
		for ( int i = 0; i < VI; ++i ) {
			const int u = tid + i * T, kp = u >> 5, c = u & 31;
			#pragma unroll
			for ( int h = 0; h < 2; ++h ) {
				rv[i][h].w = make_uint4( 0, 0, 0, 0 );
				rv[i][h].s = 0;
				if ( u < ( WM_BK / 2 ) * 32 && kb + 2 * kp + h <= max_pos ) rv[i][h] = kv_fetch8<VQ>( kv.v, kv.vs, kv_base + static_cast<size_t>( kb + 2 * kp + h ) * kv_stride + c * 8 );
			}
		}
	};
	if ( kb0 < kb1 ) fetch( kb0 );

	for ( int kb = kb0; kb < kb1; kb += WM_BK ) {
		#pragma unroll
		for ( int i = 0; i < KI; ++i ) {
			const int u = tid + i * T, key = u >> 5, c = u & 31;
			if ( u < WM_BK * 32 ) *reinterpret_cast<uint4*>( sK + key * WM_KS + c * 8 ) = kv_decode8_h<KQ>( rk[i] );
		}
		#pragma unroll
		for ( int i = 0; i < VI; ++i ) {
			const int u = tid + i * T, kp = u >> 5, c = u & 31;
			if ( u < ( WM_BK / 2 ) * 32 ) {
				const uint4 va = kv_decode8_h<VQ>( rv[i][0] ), vb = kv_decode8_h<VQ>( rv[i][1] );
				const half* v0 = reinterpret_cast<const half*>( &va );
				const half* v1 = reinterpret_cast<const half*>( &vb );
				#pragma unroll
				for ( int j = 0; j < 8; ++j ) *reinterpret_cast<half2*>( sVt + ( c * 8 + j ) * WM_VS + 2 * kp ) = __halves2half2( v0[j], v1[j] );
			}
		}
		__syncthreads();
		if ( kb + WM_BK < kb1 ) fetch( kb + WM_BK );

		if ( wave_active ) {
			// S^T (16 keys x 16 queries per tile): lane column = its query, element i = key 2i + hi
			wm_v8f s[NKT];
			#pragma unroll
			for ( int kt = 0; kt < NKT; ++kt ) s[kt] = wm_v8f{ 0, 0, 0, 0, 0, 0, 0, 0 };
			const half* qrow = sQ + ( wave * 16 + ql ) * WM_KS;
			#pragma unroll 4
			for ( int ks = 0; ks < HD / 16; ++ks ) {
				const wm_v16h b = wm_ld16( qrow + 16 * ks );
				#pragma unroll
				for ( int kt = 0; kt < NKT; ++kt ) s[kt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32( wm_ld16( sK + ( kt * 16 + ql ) * WM_KS + 16 * ks ), b, s[kt] );
			}

			float mx = -1e30f;
			#pragma unroll
			for ( int kt = 0; kt < NKT; ++kt ) {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) {
					const float v = ( kb + kt * 16 + 2 * i + hi <= spos ) ? s[kt][i] * scale2 : -1e30f;
					s[kt][i] = v;
					mx = fmaxf( mx, v );
				}
			}
			mx = fmaxf( mx, __shfl_xor( mx, 16 ) );
			const float m_new = fmaxf( m_run, mx );
			const float alpha = exp2f( m_run - m_new );
			float lsum = 0.0f;
			_Float16 ph[NKT][8];
			#pragma unroll
			for ( int kt = 0; kt < NKT; ++kt ) {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) {
					const float p = s[kt][i] > -1e29f ? exp2f( s[kt][i] - m_new ) : 0.0f;
					lsum += p;
					ph[kt][i] = static_cast<_Float16>( p );
				}
			}
			l_run = l_run * alpha + lsum;
			m_run = m_new;
			if ( __any( alpha != 1.0f ) ) {
				#pragma unroll
				for ( int d = 0; d < 16; ++d ) o[d] *= alpha;
			}

			// P^T as the B operand: this lane's 8 keys (parity hi) and the partner lane's (parity 1 - hi), interleaved to key order
			#pragma unroll
			for ( int kt = 0; kt < NKT; ++kt ) {
				wm_v16h x;
				#pragma unroll
				for ( int j = 0; j < 4; ++j ) {
					const uint32_t mine = __builtin_bit_cast( uint32_t, wm_v2h{ ph[kt][2 * j], ph[kt][2 * j + 1] } );
					const uint32_t other = __shfl_xor( mine, 16 );
					const uint32_t ev = hi ? other : mine;   // keys 4j, 4j+2 (even keys sit in the hi = 0 lane)
					const uint32_t od = hi ? mine : other;   // keys 4j+1, 4j+3
					const wm_v2h e = __builtin_bit_cast( wm_v2h, ev ), od2 = __builtin_bit_cast( wm_v2h, od );
					const _Float16 e0 = e[0], e1 = e[1], o0 = od2[0], o1 = od2[1];
					x[4 * j] = e0; x[4 * j + 1] = o0; x[4 * j + 2] = e1; x[4 * j + 3] = o1;
				}
				#pragma unroll
				for ( int dt = 0; dt < 16; ++dt ) o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32( wm_ld16( sVt + ( 16 * dt + ql ) * WM_VS + kt * 16 ), x, o[dt] );
			}
		}
		__syncthreads();
	}

	const float l_tot = l_run + __shfl_xor( l_run, 16 );

	if ( nsplit > 1 ) {
		// Unnormalized partial (m, l, O) for the merge; m in natural-log units like the chunked decode kernel
		if ( qs < q_end ) {
			float* ws = workspace + ( ( static_cast<size_t>( qs / G ) * num_q_heads + kvh * G + qs % G ) * nsplit + blockIdx.z ) * ( ATTN_MAX_HEAD_DIM + 2 );
			if ( hi == 0 ) {
				ws[0] = m_run * 0.6931471805599453f;
				ws[1] = l_tot;
			}
			#pragma unroll
			for ( int dt = 0; dt < 16; ++dt ) {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) ws[2 + 16 * dt + 2 * i + hi] = o[dt][i];
			}
		}
		return;
	}

	const float inv_l = l_tot > 0.0f ? 1.0f / l_tot : 0.0f;
	if ( wave_active ) {
		#pragma unroll
		for ( int dt = 0; dt < 16; ++dt ) {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) sO[( wave * 16 + ql ) * WM_OS + 16 * dt + 2 * i + hi] = o[dt][i] * inv_l;
		}
	}
	__syncthreads();

	if constexpr ( VQ == KVQ::Q4 ) {
		// Undo the per-64 rotation of V
		if ( tid < WM_BQ * 4 ) {
			float* x = sO + ( tid >> 2 ) * WM_OS + ( tid & 3 ) * 64;
			for ( int h = 1; h < 64; h <<= 1 ) {
				for ( int i = 0; i < 64; ++i ) {
					if ( ( i & h ) == 0 ) {
						const float a = x[i], b = x[i | h];
						x[i] = a + b;
						x[i | h] = a - b;
					}
				}
			}
			for ( int i = 0; i < 64; ++i ) x[i] *= 0.125f;
		}
		__syncthreads();
	}

	for ( int u = tid; u < WM_BQ * ( HD / 8 ); u += T ) {
		const int qq = u >> 5, c = u & 31;
		const int qi = q_base + qq;
		if ( qi >= q_end ) continue;
		const size_t off = ( static_cast<size_t>( qi / G ) * num_q_heads + kvh * G + qi % G ) * HD + c * 8;
		float sig[8];
		#pragma unroll
		for ( int i = 0; i < 8; ++i ) sig[i] = 1.0f;
		if ( gate != nullptr ) {
			const uint4 graw = *reinterpret_cast<const uint4*>( gate + off );
			const half2* g2 = reinterpret_cast<const half2*>( &graw );
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const float2 f = __half22float2( g2[i] );
				sig[2 * i] = 1.0f / ( 1.0f + __expf( -f.x ) );
				sig[2 * i + 1] = 1.0f / ( 1.0f + __expf( -f.y ) );
			}
		}
		const float* ov = sO + qq * WM_OS + c * 8;
		uint4 res;
		half2* r2 = reinterpret_cast<half2*>( &res );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) r2[i] = __floats2half2_rn( ov[2 * i] * sig[2 * i], ov[2 * i + 1] * sig[2 * i + 1] );
		*reinterpret_cast<uint4*>( out + off ) = res;
	}
}
