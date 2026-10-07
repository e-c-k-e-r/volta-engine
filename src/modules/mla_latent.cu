#include "engine/modules/mla.h"
#include <cuda_fp16.h>
#include <cmath>

namespace engine::modules {

namespace {

constexpr int LORA = MLALatent::LORA;
constexpr int HD = MLALatent::HEAD;
constexpr int ROWG = 16; // rows per pass of the projection kernels

// q_lat[r][h][c] = sum_d q[r][h][d] * W_UK[h][d][c]. Block = (head, 64-column slice), 32 threads of 2 columns; the q rows of
// a group of 16 are staged in shared memory, the weight column slice streams once per group.
__global__ __launch_bounds__( 32 )
void absorb_q_kernel( const half* __restrict__ q, const half* __restrict__ kv_b, half* __restrict__ q_lat, int heads, int rows ) {
	__shared__ float sq[ROWG][HD];
	const int h = blockIdx.x;
	const int c0 = blockIdx.y * 64 + threadIdx.x * 2;
	const half* w = kv_b + static_cast<size_t>( h ) * 2 * HD * LORA; // K rows first
	for ( int r0 = 0; r0 < rows; r0 += ROWG ) {
		const int nr = min( ROWG, rows - r0 );
		__syncthreads();
		for ( int i = threadIdx.x; i < nr * HD; i += 32 ) {
			const int r = i / HD, d = i % HD;
			sq[r][d] = __half2float( q[( static_cast<size_t>( r0 + r ) * heads + h ) * HD + d] );
		}
		__syncthreads();
		float acc[ROWG][2];
		#pragma unroll
		for ( int r = 0; r < ROWG; ++r ) { acc[r][0] = 0.0f; acc[r][1] = 0.0f; }
		for ( int d = 0; d < HD; ++d ) {
			const float2 wf = __half22float2( *reinterpret_cast<const half2*>( w + static_cast<size_t>( d ) * LORA + c0 ) );
			#pragma unroll
			for ( int r = 0; r < ROWG; ++r ) {
				const float qv = sq[r][d];
				acc[r][0] = fmaf( qv, wf.x, acc[r][0] );
				acc[r][1] = fmaf( qv, wf.y, acc[r][1] );
			}
		}
		for ( int r = 0; r < nr; ++r ) {
			*reinterpret_cast<half2*>( q_lat + ( static_cast<size_t>( r0 + r ) * heads + h ) * LORA + c0 ) = __floats2half2_rn( acc[r][0], acc[r][1] );
		}
	}
}

// out[r][h][v] = sum_c W_UV[h][v][c] * o_lat[r][h][c]. Block = (head, 64 values), one thread per value streaming its 1 KiB weight
// row once per group of 16 rows (o_lat of the group staged in shared memory).
__global__ __launch_bounds__( 64 )
void expand_out_kernel( const half* __restrict__ o_lat, const half* __restrict__ kv_b, half* __restrict__ out, int heads, int rows ) {
	__shared__ float so[ROWG][LORA];
	const int h = blockIdx.x;
	const int v = blockIdx.y * 64 + threadIdx.x;
	const half* w = kv_b + ( static_cast<size_t>( h ) * 2 * HD + HD + v ) * LORA; // V rows follow the K rows
	for ( int r0 = 0; r0 < rows; r0 += ROWG ) {
		const int nr = min( ROWG, rows - r0 );
		__syncthreads();
		for ( int i = threadIdx.x; i < nr * LORA; i += 64 ) {
			const int r = i / LORA, c = i % LORA;
			so[r][c] = __half2float( o_lat[( static_cast<size_t>( r0 + r ) * heads + h ) * LORA + c] );
		}
		__syncthreads();
		float acc[ROWG];
		#pragma unroll
		for ( int r = 0; r < ROWG; ++r ) acc[r] = 0.0f;
		for ( int c = 0; c < LORA; c += 8 ) {
			const uint4 raw = *reinterpret_cast<const uint4*>( w + c );
			const half2* w2 = reinterpret_cast<const half2*>( &raw );
			float wf[8];
			#pragma unroll
			for ( int i = 0; i < 4; ++i ) {
				const float2 f = __half22float2( w2[i] );
				wf[2 * i] = f.x;
				wf[2 * i + 1] = f.y;
			}
			#pragma unroll
			for ( int r = 0; r < ROWG; ++r ) {
				#pragma unroll
				for ( int i = 0; i < 8; ++i ) acc[r] = fmaf( wf[i], so[r][c + i], acc[r] );
			}
		}
		for ( int r = 0; r < nr; ++r ) {
			out[( static_cast<size_t>( r0 + r ) * heads + h ) * HD + v] = __float2half_rn( acc[r] );
		}
	}
}

__global__ void store_latent_kernel( half* __restrict__ cache, const half* __restrict__ c, int layer, int max_seq, int rows, pod::SlotRef slot ) {
	const int r = blockIdx.x;
	const int pos = slot.rows->pos[slot.row + r];
	const size_t off = static_cast<size_t>( slot.rows->slot[slot.row + r] ) * slot.stride + ( static_cast<size_t>( layer ) * max_seq + pos ) * LORA;
	for ( int i = threadIdx.x; i < LORA; i += blockDim.x ) cache[off + i] = c[static_cast<size_t>( r ) * LORA + i];
}

constexpr int CHUNK_KEYS = 1024;
constexpr int TILE = 16;
constexpr int CS = LORA + 8; // padded key row (halves) in shared memory

// Latent attention, one block per (row, chunk of 1024 keys), all heads of the row: a 16-key tile of c is staged once and
// serves the 16 heads' scores and their value sums. 512 threads: scores take two threads per (head, key) pair, the value
// sum one thread per latent dimension holding all 16 heads' accumulators.
template <int HEADS>
__global__ __launch_bounds__( 512 )
void latent_attn_kernel(
	const half* __restrict__ q_lat,
	const half* __restrict__ cache,
	half* __restrict__ o_lat,
	float* __restrict__ workspace,
	size_t slot_stride,
	int layer,
	int max_seq,
	const pod::BatchRows* __restrict__ rows,
	int row0,
	int n_chunks,
	float scale
) {
	extern __shared__ __align__( 16 ) unsigned char smem_raw[];
	float* sq = reinterpret_cast<float*>( smem_raw );                       // [HEADS][LORA]
	half* sc = reinterpret_cast<half*>( sq + HEADS * LORA );                // [TILE][CS]
	float* ss = reinterpret_cast<float*>( sc + TILE * CS );                 // [HEADS][TILE] scores, then probabilities
	float* sal = ss + HEADS * TILE;                                         // [HEADS] rescale factors
	const int m = blockIdx.x, chunk = blockIdx.y, t = threadIdx.x;
	const int pos = rows->pos[row0 + m];
	const int total = min( pos + 1, max_seq );
	const int kv_start = chunk * CHUNK_KEYS;
	const int kv_end = min( kv_start + CHUNK_KEYS, total );
	const size_t base = static_cast<size_t>( rows->slot[row0 + m] ) * slot_stride + static_cast<size_t>( layer ) * max_seq * LORA;

	float o[HEADS];
	#pragma unroll
	for ( int h = 0; h < HEADS; ++h ) o[h] = 0.0f;
	float m_run = -1e30f, l_run = 0.0f; // meaningful on t < HEADS

	if ( kv_start < total ) {
		for ( int i = t; i < HEADS * LORA; i += 512 ) sq[i] = __half2float( q_lat[( static_cast<size_t>( m ) * HEADS ) * LORA + i] );
		for ( int kb = kv_start; kb < kv_end; kb += TILE ) {
			const int nk = min( TILE, kv_end - kb );
			__syncthreads();
			for ( int u = t; u < TILE * ( LORA / 8 ); u += 512 ) {
				const int k = u / ( LORA / 8 ), c8 = u % ( LORA / 8 );
				uint4 raw = make_uint4( 0, 0, 0, 0 );
				if ( k < nk ) raw = *reinterpret_cast<const uint4*>( cache + base + static_cast<size_t>( kb + k ) * LORA + c8 * 8 );
				*reinterpret_cast<uint4*>( sc + k * CS + c8 * 8 ) = raw;
			}
			__syncthreads();
			// Scores: pair p = (head p / TILE, key p % TILE), lanes 2p and 2p + 1 take half of the dimensions each
			{
				const int p = t >> 1, half_sel = t & 1;
				const int h = p / TILE, k = p % TILE;
				const float* qh = sq + h * LORA + half_sel * 256;
				const half* ck = sc + k * CS + half_sel * 256;
				float dot = 0.0f;
				#pragma unroll 4
				for ( int d = 0; d < 256; d += 8 ) {
					const uint4 craw = *reinterpret_cast<const uint4*>( ck + d );
					const half2* c2 = reinterpret_cast<const half2*>( &craw );
					const float4 qa = *reinterpret_cast<const float4*>( qh + d );
					const float4 qb = *reinterpret_cast<const float4*>( qh + d + 4 );
					const float2 f0 = __half22float2( c2[0] ), f1 = __half22float2( c2[1] ), f2 = __half22float2( c2[2] ), f3 = __half22float2( c2[3] );
					dot = fmaf( qa.x, f0.x, dot ); dot = fmaf( qa.y, f0.y, dot );
					dot = fmaf( qa.z, f1.x, dot ); dot = fmaf( qa.w, f1.y, dot );
					dot = fmaf( qb.x, f2.x, dot ); dot = fmaf( qb.y, f2.y, dot );
					dot = fmaf( qb.z, f3.x, dot ); dot = fmaf( qb.w, f3.y, dot );
				}
				dot += __shfl_xor_sync( 0xffffffff, dot, 1 );
				if ( half_sel == 0 ) ss[h * TILE + k] = ( k < nk ) ? dot * scale : -1e30f;
			}
			__syncthreads();
			if ( t < HEADS ) {
				float tile_max = -1e30f;
				#pragma unroll
				for ( int k = 0; k < TILE; ++k ) tile_max = fmaxf( tile_max, ss[t * TILE + k] );
				const float m_new = fmaxf( m_run, tile_max );
				const float alpha = __expf( m_run - m_new );
				float lsum = 0.0f;
				#pragma unroll
				for ( int k = 0; k < TILE; ++k ) {
					const float pk = ( k < nk ) ? __expf( ss[t * TILE + k] - m_new ) : 0.0f;
					ss[t * TILE + k] = pk;
					lsum += pk;
				}
				l_run = l_run * alpha + lsum;
				m_run = m_new;
				sal[t] = alpha;
			}
			__syncthreads();
			#pragma unroll
			for ( int h = 0; h < HEADS; ++h ) o[h] *= sal[h];
			for ( int k = 0; k < nk; ++k ) {
				const float cv = __half2float( sc[k * CS + t] );
				#pragma unroll
				for ( int h = 0; h < HEADS; ++h ) o[h] = fmaf( ss[h * TILE + k], cv, o[h] );
			}
		}
	}

	if ( n_chunks == 1 ) {
		// Row's whole context in one chunk: write the normalized result
		__shared__ float s_inv[HEADS];
		if ( t < HEADS ) s_inv[t] = l_run > 0.0f ? 1.0f / l_run : 0.0f;
		__syncthreads();
		#pragma unroll
		for ( int h = 0; h < HEADS; ++h ) o_lat[( static_cast<size_t>( m ) * HEADS + h ) * LORA + t] = __float2half_rn( o[h] * s_inv[h] );
	}
	else {
		#pragma unroll
		for ( int h = 0; h < HEADS; ++h ) {
			float* ws = workspace + ( ( static_cast<size_t>( m ) * HEADS + h ) * n_chunks + chunk ) * ( LORA + 2 );
			ws[2 + t] = o[h];
			if ( t == h ) {
				ws[0] = m_run;
				ws[1] = l_run;
			}
		}
	}
}

// Merge a row's chunk partials: out = sum_c exp(m_c - m) o_c / sum_c exp(m_c - m) l_c; empty chunks (l = 0) are skipped.
__global__ __launch_bounds__( 512 )
void latent_reduce_kernel( const float* __restrict__ workspace, half* __restrict__ o_lat, int heads, int n_chunks ) {
	const int m = blockIdx.y, h = blockIdx.x, t = threadIdx.x;
	const float* ws = workspace + ( static_cast<size_t>( m ) * heads + h ) * n_chunks * ( LORA + 2 );
	float mx = -1e30f;
	for ( int c = 0; c < n_chunks; ++c ) {
		if ( ws[c * ( LORA + 2 ) + 1] > 0.0f ) mx = fmaxf( mx, ws[c * ( LORA + 2 )] );
	}
	float l = 0.0f, acc = 0.0f;
	for ( int c = 0; c < n_chunks; ++c ) {
		const float lc = ws[c * ( LORA + 2 ) + 1];
		if ( lc <= 0.0f ) continue;
		const float w = __expf( ws[c * ( LORA + 2 )] - mx );
		l += w * lc;
		acc += w * ws[c * ( LORA + 2 ) + 2 + t];
	}
	o_lat[( static_cast<size_t>( m ) * heads + h ) * LORA + t] = __float2half_rn( l > 0.0f ? acc / l : 0.0f );
}

} // namespace

void MLALatent::absorb_q( const pod::TensorView& q, const pod::TensorView& kv_b, pod::TensorView& q_lat, int heads, int rows, cudaStream_t stream ) {
	absorb_q_kernel<<<dim3( heads, LORA / 64 ), dim3( 32 ), 0, stream>>>( q.ptr<const half>(), kv_b.ptr<const half>(), q_lat.ptr<half>(), heads, rows );
}

void MLALatent::expand_out( const pod::TensorView& o_lat, const pod::TensorView& kv_b, pod::TensorView& out, int heads, int rows, cudaStream_t stream ) {
	expand_out_kernel<<<dim3( heads, HEAD / 64 ), dim3( 64 ), 0, stream>>>( o_lat.ptr<const half>(), kv_b.ptr<const half>(), out.ptr<half>(), heads, rows );
}

void MLALatent::store( half* cache, const pod::TensorView& c, int layer, int max_seq, int rows, const pod::SlotRef& slot, cudaStream_t stream ) {
	ENGINE_ASSERT( slot.rows != nullptr, "MLALatent::store: device-resolved positions only" );
	store_latent_kernel<<<dim3( rows ), dim3( 128 ), 0, stream>>>( cache, c.ptr<const half>(), layer, max_seq, rows, slot );
}

void MLALatent::attend( const half* cache, const pod::TensorView& q_lat, pod::TensorView& o_lat, int layer, int max_seq, int rows, int heads,
	const pod::SlotRef& slot, float* workspace, int key_hint, cudaStream_t stream ) {
	(void) key_hint;
	ENGINE_ASSERT( heads == 16 && slot.rows != nullptr, "MLALatent::attend: 16 heads per rank, device-resolved positions" );
	const int n_chunks = workspace_chunks( max_seq );
	const float scale = 1.0f / sqrtf( static_cast<float>( HEAD ) );
	const size_t smem = static_cast<size_t>( 16 ) * LORA * 4 + static_cast<size_t>( TILE ) * CS * 2 + ( 16 * TILE + 16 ) * 4;
	static bool attr[ENGINE_NUM_GPUS] = {};
	int device = 0;
	cudaGetDevice( &device );
	if ( !attr[device] ) {
		CUDA_CHECK( cudaFuncSetAttribute( latent_attn_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>( smem ) ) );
		attr[device] = true;
	}
	for ( int row0 = slot.row; row0 < slot.row + rows; row0 += pod::MAX_BATCH ) {
		const int M = min( pod::MAX_BATCH, slot.row + rows - row0 );
		const size_t off = static_cast<size_t>( row0 - slot.row ) * heads * LORA;
		latent_attn_kernel<16><<<dim3( M, n_chunks ), dim3( 512 ), smem, stream>>>(
			q_lat.ptr<const half>() + off, cache, o_lat.ptr<half>() + off, workspace, slot.stride, layer, max_seq, slot.rows, row0, n_chunks, scale );
		if ( n_chunks > 1 ) {
			latent_reduce_kernel<<<dim3( heads, M ), dim3( 512 ), 0, stream>>>( workspace, o_lat.ptr<half>() + off, heads, n_chunks );
		}
	}
}

} // namespace engine::modules
