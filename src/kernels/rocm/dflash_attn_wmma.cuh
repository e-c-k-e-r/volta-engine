#pragma once
// ROCm only (RDNA3, wave32). Included by src/spec/dflash_kernels.cu for dflash_attn: the drafter's attention over its context ring and the
// block's own rows (no mask, every row sees every key) on the wave32 WMMA, laid out like attn_wmma.cuh (S^T = K Q^T so each lane owns one
// query column and the softmax is in-register, P^T feeds O^T = V^T P^T, V staged transposed, the next tile fetched into registers during
// compute). A block owns one kv head and 16 * DFW_W query slots (row, head-in-group pairs) and a split of the keys; with several splits the
// partials (m, l, O) are merged by dflash_attn_wmma_reduce. head_dim 128 or 256, FP16 caches.
#include "wmma.cuh"

#define DFW_W 3
#define DFW_BQ ( 16 * DFW_W )

template <int HD> struct DfwTile { static constexpr int BK = HD <= 128 ? 32 : 16; };

template <int HD> constexpr size_t dfw_smem_bytes() {
	constexpr int BK = DfwTile<HD>::BK;
	constexpr size_t tiles = ( (size_t) DFW_BQ * ( HD + 8 ) + (size_t) BK * ( HD + 8 ) + (size_t) HD * ( BK + 8 ) ) * 2;
	constexpr size_t out = (size_t) DFW_BQ * ( HD + 1 ) * 4;   // the output staging reuses the tiles
	return tiles > out ? tiles : out;
}

template <int HD>
__global__ __launch_bounds__( 32 * DFW_W )
void dflash_attn_wmma_kernel(
	const half* __restrict__ q,        // [rows, q_heads, HD]
	const half* __restrict__ kcache,   // [slots, layers, kv_heads, HD]
	const half* __restrict__ vcache,
	half* __restrict__ out,            // [rows, q_heads, HD]
	float* __restrict__ ws,            // nsplit > 1: [rows * q_heads][nsplit][HD + 2] unnormalized partials (m, l, O)
	int rows,
	int q_heads,
	int kv_heads,
	int nctx,
	int first_ctx,
	int ring_slots,
	int ctx_layer,
	int stride_slot,
	int stride_layer,
	int stride_head,
	int own_slot,
	int own_first_row,
	float scale,
	int nsplit
) {
	constexpr int BK = DfwTile<HD>::BK;
	constexpr int T = 32 * DFW_W;
	constexpr int KS = HD + 8, VS = BK + 8, NDT = HD / 16, NKT = BK / 16, CPR = HD / 8;   // chunks of 8 halves per row
	extern __shared__ __align__( 16 ) unsigned char dfw_smem[];
	half* sQ = reinterpret_cast<half*>( dfw_smem );
	half* sK = sQ + DFW_BQ * KS;
	half* sVt = sK + BK * KS;
	float* sO = reinterpret_cast<float*>( dfw_smem );

	const int G = q_heads / kv_heads;
	const int kvh = blockIdx.x;
	const int q_base = blockIdx.y * DFW_BQ;
	const int q_end = min( q_base + DFW_BQ, rows * G );
	const int tid = threadIdx.x;
	const int wave = tid >> 5;
	const int lane = tid & 31;
	const int ql = lane & 15;
	const int hi = lane >> 4;
	const int nkeys = nctx + ( rows - own_first_row );

	// Key kk is a context slot first (a ring), then one of the block's own rows
	auto row_ptr = [&]( const half* base, int kk ) {
		int slot;
		if ( kk < nctx ) {
			slot = first_ctx + kk;
			if ( slot >= ring_slots ) slot -= ring_slots;
		}
		else slot = own_slot + own_first_row + ( kk - nctx );
		return base + static_cast<int64_t>( slot ) * stride_slot + static_cast<int64_t>( ctx_layer ) * stride_layer + static_cast<int64_t>( kvh ) * stride_head;
	};

	const int qs = q_base + wave * 16 + ql;   // this lane's query slot (its WMMA column)
	const bool wave_active = q_base + wave * 16 < q_end;
	const int spos = qs < q_end ? nkeys - 1 : -1;

	for ( int u = tid; u < DFW_BQ * CPR; u += T ) {
		const int qq = u / CPR, c = u % CPR;
		const int qi = q_base + qq;
		uint4 raw = make_uint4( 0, 0, 0, 0 );
		if ( qi < q_end ) raw = *reinterpret_cast<const uint4*>( q + ( static_cast<int64_t>( qi / G ) * q_heads + kvh * G + qi % G ) * HD + c * 8 );
		*reinterpret_cast<uint4*>( sQ + qq * KS + c * 8 ) = raw;
	}

	wm_v8f o[NDT];
	#pragma unroll
	for ( int d = 0; d < NDT; ++d ) o[d] = wm_v8f{ 0, 0, 0, 0, 0, 0, 0, 0 };
	float m_run = -1e30f, l_run = 0.0f;
	const float scale2 = scale * 1.4426950408889634f;

	const int ntiles = ( nkeys + BK - 1 ) / BK;
	const int kb0 = ( blockIdx.z * ntiles / nsplit ) * BK;
	const int kb1 = ( ( blockIdx.z + 1 ) * ntiles / nsplit ) * BK;

	constexpr int KI = ( BK * CPR + T - 1 ) / T;
	constexpr int VI = ( ( BK / 2 ) * CPR + T - 1 ) / T;
	uint4 rk[KI], rv[VI][2];
	auto fetch = [&]( int kb ) {
		#pragma unroll
		for ( int i = 0; i < KI; ++i ) {
			const int u = tid + i * T, key = u / CPR, c = u % CPR;
			rk[i] = make_uint4( 0, 0, 0, 0 );
			if ( u < BK * CPR && kb + key < nkeys ) rk[i] = *reinterpret_cast<const uint4*>( row_ptr( kcache, kb + key ) + c * 8 );
		}
		#pragma unroll
		for ( int i = 0; i < VI; ++i ) {
			const int u = tid + i * T, kp = u / CPR, c = u % CPR;
			#pragma unroll
			for ( int h = 0; h < 2; ++h ) {
				rv[i][h] = make_uint4( 0, 0, 0, 0 );
				if ( u < ( BK / 2 ) * CPR && kb + 2 * kp + h < nkeys ) rv[i][h] = *reinterpret_cast<const uint4*>( row_ptr( vcache, kb + 2 * kp + h ) + c * 8 );
			}
		}
	};
	if ( kb0 < kb1 ) fetch( kb0 );

	for ( int kb = kb0; kb < kb1; kb += BK ) {
		#pragma unroll
		for ( int i = 0; i < KI; ++i ) {
			const int u = tid + i * T, key = u / CPR, c = u % CPR;
			if ( u < BK * CPR ) *reinterpret_cast<uint4*>( sK + key * KS + c * 8 ) = rk[i];
		}
		#pragma unroll
		for ( int i = 0; i < VI; ++i ) {
			const int u = tid + i * T, kp = u / CPR, c = u % CPR;
			if ( u < ( BK / 2 ) * CPR ) {
				const half* v0 = reinterpret_cast<const half*>( &rv[i][0] );
				const half* v1 = reinterpret_cast<const half*>( &rv[i][1] );
				#pragma unroll
				for ( int j = 0; j < 8; ++j ) *reinterpret_cast<half2*>( sVt + ( c * 8 + j ) * VS + 2 * kp ) = __halves2half2( v0[j], v1[j] );
			}
		}
		__syncthreads();
		if ( kb + BK < kb1 ) fetch( kb + BK );

		if ( wave_active ) {
			wm_v8f s[NKT];
			#pragma unroll
			for ( int kt = 0; kt < NKT; ++kt ) s[kt] = wm_v8f{ 0, 0, 0, 0, 0, 0, 0, 0 };
			const half* qrow = sQ + ( wave * 16 + ql ) * KS;
			#pragma unroll 4
			for ( int ks = 0; ks < HD / 16; ++ks ) {
				const wm_v16h b = wm_ld16( qrow + 16 * ks );
				#pragma unroll
				for ( int kt = 0; kt < NKT; ++kt ) s[kt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32( wm_ld16( sK + ( kt * 16 + ql ) * KS + 16 * ks ), b, s[kt] );
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
				for ( int d = 0; d < NDT; ++d ) o[d] *= alpha;
			}

			// P^T as the B operand: this lane's 8 keys (parity hi) and the partner lane's, interleaved back to key order
			#pragma unroll
			for ( int kt = 0; kt < NKT; ++kt ) {
				wm_v16h x;
				#pragma unroll
				for ( int j = 0; j < 4; ++j ) {
					const uint32_t mine = __builtin_bit_cast( uint32_t, wm_v2h{ ph[kt][2 * j], ph[kt][2 * j + 1] } );
					const uint32_t other = __shfl_xor( mine, 16 );
					const uint32_t ev = hi ? other : mine;
					const uint32_t od = hi ? mine : other;
					const wm_v2h e = __builtin_bit_cast( wm_v2h, ev ), od2 = __builtin_bit_cast( wm_v2h, od );
					x[4 * j] = e[0]; x[4 * j + 1] = od2[0]; x[4 * j + 2] = e[1]; x[4 * j + 3] = od2[1];
				}
				#pragma unroll
				for ( int dt = 0; dt < NDT; ++dt ) o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32( wm_ld16( sVt + ( 16 * dt + ql ) * VS + kt * 16 ), x, o[dt] );
			}
		}
		__syncthreads();
	}

	const float l_tot = l_run + __shfl_xor( l_run, 16 );

	if ( nsplit > 1 ) {
		if ( qs < q_end ) {
			float* p = ws + ( ( static_cast<int64_t>( qs / G ) * q_heads + kvh * G + qs % G ) * nsplit + blockIdx.z ) * ( HD + 2 );
			if ( hi == 0 ) {
				p[0] = m_run * 0.6931471805599453f;   // natural-log units
				p[1] = l_tot;
			}
			#pragma unroll
			for ( int dt = 0; dt < NDT; ++dt ) {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) p[2 + 16 * dt + 2 * i + hi] = o[dt][i];
			}
		}
		return;
	}

	const float inv_l = l_tot > 0.0f ? 1.0f / l_tot : 0.0f;
	if ( wave_active ) {
		#pragma unroll
		for ( int dt = 0; dt < NDT; ++dt ) {
			#pragma unroll
			for ( int i = 0; i < 8; ++i ) sO[( wave * 16 + ql ) * ( HD + 1 ) + 16 * dt + 2 * i + hi] = o[dt][i] * inv_l;
		}
	}
	__syncthreads();
	for ( int u = tid; u < DFW_BQ * CPR; u += T ) {
		const int qq = u / CPR, c = u % CPR;
		const int qi = q_base + qq;
		if ( qi >= q_end ) continue;
		const float* ov = sO + qq * ( HD + 1 ) + c * 8;
		uint4 res;
		half2* r2 = reinterpret_cast<half2*>( &res );
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) r2[i] = __floats2half2_rn( ov[2 * i], ov[2 * i + 1] );
		*reinterpret_cast<uint4*>( out + ( static_cast<int64_t>( qi / G ) * q_heads + kvh * G + qi % G ) * HD + c * 8 ) = res;
	}
}

// Merges the nsplit partials of one (row, head): grid (q_heads, rows), HD threads
template <int HD>
__global__ void dflash_attn_wmma_reduce( const float* __restrict__ ws, half* __restrict__ out, int q_heads, int nsplit ) {
	const int head = blockIdx.x, row = blockIdx.y, d = threadIdx.x;
	const float* base = ws + ( static_cast<int64_t>( row ) * q_heads + head ) * nsplit * ( HD + 2 );
	float m_acc = -1e30f, l_acc = 0.0f, o_acc = 0.0f;
	for ( int c = 0; c < nsplit; ++c ) {
		const float* p = base + static_cast<int64_t>( c ) * ( HD + 2 );
		const float m_c = p[0], l_c = p[1];
		if ( l_c == 0.0f ) continue;
		const float m_new = fmaxf( m_acc, m_c );
		const float alpha = __expf( m_acc - m_new ), beta = __expf( m_c - m_new );
		o_acc = fmaf( alpha, o_acc, beta * p[2 + d] );
		l_acc = fmaf( alpha, l_acc, beta * l_c );
		m_acc = m_new;
	}
	out[( static_cast<int64_t>( row ) * q_heads + head ) * HD + d] = __float2half_rn( l_acc > 0.0f ? o_acc / l_acc : 0.0f );
}

// Partial-result workspace: rows * q_heads * nsplit * (HD + 2) floats, at most 16 rows x 64 heads x 32 splits at head_dim 256
__device__ float g_dfw_workspace[16 * 64 * 32 * 258];

inline float* dfw_workspace_ptr() {
	static float* ptrs[ENGINE_NUM_GPUS] = {};
	int dev = 0;
	CUDA_CHECK( cudaGetDevice( &dev ) );
	ENGINE_ASSERT( dev >= 0 && dev < ENGINE_NUM_GPUS, "dfw_workspace_ptr: device out of range" );
	if ( !ptrs[dev] ) {
		void* p = nullptr;
		CUDA_CHECK( cudaGetSymbolAddress( &p, g_dfw_workspace ) );
		ptrs[dev] = static_cast<float*>( p );
	}
	return ptrs[dev];
}

// Launches the kernel (and the merge); false when the shape is outside what it handles
template <int HD>
inline void dflash_attn_wmma_launch(
	const half* q, const half* kcache, const half* vcache, half* out,
	int rows, int q_heads, int kv_heads,
	int nctx, int first_ctx, int ring_slots, int ctx_layer,
	int stride_slot, int stride_layer, int stride_head, int own_slot, int own_first_row, cudaStream_t stream
) {
	constexpr int BK = DfwTile<HD>::BK;
	const int G = q_heads / kv_heads;
	const int qtiles = ( rows * G + DFW_BQ - 1 ) / DFW_BQ;
	const int base_blocks = kv_heads * qtiles;
	const int nkeys = nctx + ( rows - own_first_row );
	const int ntiles = ( nkeys + BK - 1 ) / BK;
	int nsplit = 1;
	if ( base_blocks < 96 ) nsplit = MIN( ( 192 + base_blocks - 1 ) / base_blocks, 32 );
	nsplit = MAX( 1, MIN( nsplit, ntiles / 2 ) );
	ENGINE_ASSERT( static_cast<size_t>( rows ) * q_heads * nsplit <= 16 * 64 * 32 && HD + 2 <= 258, "dflash_attn_wmma: workspace" );
	float* ws = dfw_workspace_ptr();
	constexpr size_t smem = dfw_smem_bytes<HD>();
	CUDA_CHECK( cudaFuncSetAttribute( dflash_attn_wmma_kernel<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem ) );
	dflash_attn_wmma_kernel<HD><<<dim3( kv_heads, qtiles, nsplit ), 32 * DFW_W, smem, stream>>>(
		q, kcache, vcache, out, ws, rows, q_heads, kv_heads, nctx, first_ctx, ring_slots, ctx_layer,
		stride_slot, stride_layer, stride_head, own_slot, own_first_row, 1.0f / sqrtf( static_cast<float>( HD ) ), nsplit );
	if ( nsplit > 1 ) dflash_attn_wmma_reduce<HD><<<dim3( q_heads, rows ), HD, 0, stream>>>( ws, out, q_heads, nsplit );
}
