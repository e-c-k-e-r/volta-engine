#include "engine/modules/mhc.h"
#include "engine/core/batch.h"
#include <cuda_fp16.h>
#include <cmath>
#include <type_traits>

namespace engine::modules {

namespace {

#define NUM_THREADS 256
#define NUM_THREADS_A 64

// Per-row partial sums: up to MAX_ROWS rows x 64 chunks x (M + 1) values
__device__ float g_mhc_partials[pod::MAX_ROWS * 64 * 25];

__device__ __forceinline__ float sigmoidf_( float x ) {
	return 1.0f / ( 1.0f + __expf( -x ) );
}

__global__ void init_streams_kernel(
	const half* __restrict__ in,
	float* __restrict__ streams,
	int hidden_size
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ( idx >= hidden_size ) return;
	const int r = blockIdx.y;
	float val = __half2float( in[static_cast<size_t>( r ) * hidden_size + idx] );
	float* out = streams + static_cast<size_t>( r ) * 4 * hidden_size;
	#pragma unroll
	for ( int h = 0; h < 4; ++h ) {
		out[h * hidden_size + idx] = val;
	}
}

template <int H, int M_, typename FN_T>
__global__ __launch_bounds__( NUM_THREADS_A )
void hc_mix_partials_kernel(
	const float* __restrict__ streams,
	const FN_T* __restrict__ fn,
	const int row_len,
	const int chunk_cols
) {
	constexpr int M = M_;
	const int r = blockIdx.y;
	const int c0 = blockIdx.x * chunk_cols;
	const int c1 = min( c0 + chunk_cols, row_len );

	const float4* s4 = (const float4*)( streams + static_cast<size_t>( r ) * row_len );
	const int row_len4 = row_len / 4;

	float acc[M + 1];
	#pragma unroll
	for ( int k = 0; k <= M; ++k ) acc[k] = 0.0f;

	for ( int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A ) {
		float4 s = s4[c];
		acc[M] = fmaf( s.x, s.x, acc[M] );
		acc[M] = fmaf( s.y, s.y, acc[M] );
		acc[M] = fmaf( s.z, s.z, acc[M] );
		acc[M] = fmaf( s.w, s.w, acc[M] );
		#pragma unroll
		for ( int j = 0; j < M; ++j ) {
			float4 w;
			if constexpr ( std::is_same_v<FN_T, half> ) {
				int2 pk = ((const int2*) fn)[static_cast<size_t>( j ) * row_len4 + c];
				float2 lo = __half22float2( *(const half2*) &pk.x );
				float2 hi = __half22float2( *(const half2*) &pk.y );
				w = make_float4( lo.x, lo.y, hi.x, hi.y );
			}
			else {
				w = ((const float4*) fn)[static_cast<size_t>( j ) * row_len4 + c];
			}
			float d = fmaf( s.x, w.x, fmaf( s.y, w.y, fmaf( s.z, w.z, s.w * w.w ) ) );
			acc[j] += d;
		}
	}

	__shared__ float red[NUM_THREADS_A / 32][M + 1];
	int lane = threadIdx.x % 32;
	int warp = threadIdx.x / 32;
	#pragma unroll
	for ( int k = 0; k <= M; ++k ) {
		float v = acc[k];
		for ( int offset = 16; offset > 0; offset >>= 1 ) {
			v += __shfl_down_sync( 0xffffffffu, v, offset );
		}
		if ( lane == 0 ) red[warp][k] = v;
	}
	__syncthreads();

	if ( warp == 0 ) {
		float* out = g_mhc_partials + ( static_cast<size_t>( r ) * gridDim.x + blockIdx.x ) * ( M + 1 );
		for ( int k = lane; k <= M; k += 32 ) {
			float v = 0.0f;
			#pragma unroll
			for ( int w = 0; w < NUM_THREADS_A / 32; ++w ) {
				v += red[w][k];
			}
			out[k] = v;
		}
	}
}

template <int H, int M_, bool HEAD, bool HALF_OUT>
__global__ __launch_bounds__( NUM_THREADS )
void hc_mix_finalize_kernel(
	const float* __restrict__ streams,
	const float* __restrict__ base,
	const float* __restrict__ scale,
	float* __restrict__ post,
	float* __restrict__ comb,
	void* __restrict__ collapsed,
	float* __restrict__ inv_rms_buf,
	float* __restrict__ mix_logits_buf,
	float* __restrict__ pre_buf,
	const int D,
	const int chunksA,
	const int chunk_cols_c,
	const float rms_eps,
	const float hc_eps,
	const int sinkhorn_iters
) {
	constexpr int M = M_;
	const int r = blockIdx.y;
	const int row_len = H * D;

	__shared__ float mix_s[M + 1];
	__shared__ float pre_s[H];
	__shared__ float red_s[NUM_THREADS / 32][M + 1];
	const int lane = threadIdx.x % 32;
	const int warp = threadIdx.x / 32;
	if ( lane <= M ) {
		const float* p = g_mhc_partials + static_cast<size_t>( r ) * chunksA * ( M + 1 ) + lane;
		float v = 0.0f;
		for ( int i = warp; i < chunksA; i += NUM_THREADS / 32 ) {
			v += p[static_cast<size_t>( i ) * ( M + 1 )];
		}
		red_s[warp][lane] = v;
	}
	__syncthreads();
	if ( threadIdx.x <= M ) {
		float v = 0.0f;
		#pragma unroll
		for ( int w = 0; w < NUM_THREADS / 32; ++w ) {
			v += red_s[w][threadIdx.x];
		}
		mix_s[threadIdx.x] = v;
	}
	__syncthreads();
	float rmr = rsqrtf( mix_s[M] / static_cast<float>( row_len ) + rms_eps );
	if ( threadIdx.x < H ) {
		pre_s[threadIdx.x] = sigmoidf_( fmaf( mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x] ) ) + hc_eps;
	}
	__syncthreads();

	if ( inv_rms_buf && threadIdx.x == 0 && blockIdx.x == 0 ) inv_rms_buf[r] = rmr;
	if ( mix_logits_buf && threadIdx.x < M && blockIdx.x == 0 ) mix_logits_buf[r * M + threadIdx.x] = mix_s[threadIdx.x];
	if ( pre_buf && threadIdx.x < H && blockIdx.x == 0 ) pre_buf[r * H + threadIdx.x] = pre_s[threadIdx.x];

	const bool sink_warp = !HEAD && blockIdx.x == 0 && threadIdx.x < 32;
	if ( sink_warp ) {
		if ( threadIdx.x < H ) {
			post[static_cast<size_t>( r ) * H + threadIdx.x] =
				2.0f * sigmoidf_( fmaf( mix_s[H + threadIdx.x] * rmr, scale[1], base[H + threadIdx.x] ) );
		}

		if ( threadIdx.x < H * H ) {
			const unsigned mask = ( H * H == 32 ) ? 0xffffffffu : ( ( 1u << ( H * H ) ) - 1u );
			float v = fmaf( mix_s[2 * H + threadIdx.x] * rmr, scale[2], base[2 * H + threadIdx.x] );

			// softmax over rows
			float m = v;
			#pragma unroll
			for ( int o = 1; o < H; o <<= 1 ) m = fmaxf( m, __shfl_xor_sync( mask, m, o ) );
			v = __expf( v - m );
			float s = v;
			#pragma unroll
			for ( int o = 1; o < H; o <<= 1 ) s += __shfl_xor_sync( mask, s, o );
			v = __fdividef( v, s ) + hc_eps;

			// column normalize, then (iters - 1) x (row, column)
			float cs = v;
			#pragma unroll
			for ( int o = H; o < H * H; o <<= 1 ) cs += __shfl_xor_sync( mask, cs, o );
			v = __fdividef( v, cs + hc_eps );
			for ( int it = 0; it < sinkhorn_iters - 1; ++it ) {
				float rs = v;
				#pragma unroll
				for ( int o = 1; o < H; o <<= 1 ) rs += __shfl_xor_sync( mask, rs, o );
				v = __fdividef( v, rs + hc_eps );
				cs = v;
				#pragma unroll
				for ( int o = H; o < H * H; o <<= 1 ) cs += __shfl_xor_sync( mask, cs, o );
				v = __fdividef( v, cs + hc_eps );
			}
			comb[static_cast<size_t>( r ) * H * H + threadIdx.x] = v;
		}
		return;
	}

	float pre_r[H];
	#pragma unroll
	for ( int h = 0; h < H; ++h ) pre_r[h] = pre_s[h];

	const bool shrunk = !HEAD && blockIdx.x == 0;
	const int tid = shrunk ? threadIdx.x - 32 : threadIdx.x;
	const int nth = shrunk ? NUM_THREADS - 32 : NUM_THREADS;
	const int c0 = blockIdx.x * chunk_cols_c;
	const int c1 = min( c0 + chunk_cols_c, D );
	const float4* s4 = (const float4*)( streams + static_cast<size_t>( r ) * row_len );
	const int D4 = D / 4;
	for ( int c = c0 / 4 + tid; c < c1 / 4; c += nth ) {
		float4 o = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
		#pragma unroll
		for ( int h = 0; h < H; ++h ) {
			float4 s = s4[static_cast<size_t>( h ) * D4 + c];
			o.x = fmaf( pre_r[h], s.x, o.x );
			o.y = fmaf( pre_r[h], s.y, o.y );
			o.z = fmaf( pre_r[h], s.z, o.z );
			o.w = fmaf( pre_r[h], s.w, o.w );
		}
		if ( HALF_OUT ) {
			half2* out2 = (half2*)( static_cast<half*>( collapsed ) + static_cast<size_t>( r ) * D );
			out2[c * 2] = __floats2half2_rn( o.x, o.y );
			out2[c * 2 + 1] = __floats2half2_rn( o.z, o.w );
		}
		else {
			((float4*)( static_cast<float*>( collapsed ) + static_cast<size_t>( r ) * D ))[c] = o;
		}
	}
}

template <int H, typename Y_T, bool HAS_COMB>
__global__ __launch_bounds__( NUM_THREADS )
void hc_apply_kernel(
	float* __restrict__ x,
	const Y_T* __restrict__ y,
	const float* __restrict__ post,
	const float* __restrict__ comb,
	const int D,
	const int chunk_cols
) {
	const int r = blockIdx.y;
	const int c0 = blockIdx.x * chunk_cols;
	const int c1 = min( c0 + chunk_cols, D );
	const int D4 = D / 4;

	float post_r[H];
	float comb_r[H][H];
	#pragma unroll
	for ( int h = 0; h < H; ++h ) {
		post_r[h] = __ldg( post + static_cast<size_t>( r ) * H + h );
		if ( HAS_COMB ) {
			#pragma unroll
			for ( int g = 0; g < H; ++g ) {
				comb_r[h][g] = __ldg( comb + ( static_cast<size_t>( r ) * H + h ) * H + g );
			}
		}
	}

	float4* x4 = (float4*)( x + static_cast<size_t>( r ) * H * D );
	for ( int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS ) {
		float4 xv[H];
		#pragma unroll
		for ( int h = 0; h < H; ++h ) {
			xv[h] = x4[static_cast<size_t>( h ) * D4 + c];
		}

		float4 yv;
		if constexpr ( std::is_same_v<Y_T, half> ) {
			half2 y01 = ((const half2*)( y + static_cast<size_t>( r ) * D ))[c * 2];
			half2 y23 = ((const half2*)( y + static_cast<size_t>( r ) * D ))[c * 2 + 1];
			float2 lo = __half22float2( y01 );
			float2 hi = __half22float2( y23 );
			yv = make_float4( lo.x, lo.y, hi.x, hi.y );
		}
		else {
			yv = ((const float4*)( y + static_cast<size_t>( r ) * D ))[c];
		}

		#pragma unroll
		for ( int h = 0; h < H; ++h ) {
			float4 o;
			o.x = post_r[h] * yv.x;
			o.y = post_r[h] * yv.y;
			o.z = post_r[h] * yv.z;
			o.w = post_r[h] * yv.w;
			if ( HAS_COMB ) {
				#pragma unroll
				for ( int g = 0; g < H; ++g ) {
					o.x = fmaf( comb_r[g][h], xv[g].x, o.x );
					o.y = fmaf( comb_r[g][h], xv[g].y, o.y );
					o.z = fmaf( comb_r[g][h], xv[g].z, o.z );
					o.w = fmaf( comb_r[g][h], xv[g].w, o.w );
				}
			}
			else {
				o.x += xv[h].x;
				o.y += xv[h].y;
				o.z += xv[h].z;
				o.w += xv[h].w;
			}
			x4[static_cast<size_t>( h ) * D4 + c] = o;
		}
	}
}

__global__ void mhc_hyperhead_mean_kernel(
	const float* __restrict__ streams,
	half* __restrict__ final_hidden,
	int hidden_size
) {
	int idx = ( blockIdx.x * blockDim.x + threadIdx.x ) * 4;
	if ( idx < hidden_size ) {
		const float4* s4 = (const float4*)( streams + static_cast<size_t>( blockIdx.y ) * 4 * hidden_size );
		final_hidden += static_cast<size_t>( blockIdx.y ) * hidden_size;
		int D4 = hidden_size / 4;
		int c = idx / 4;
		float4 o = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
		#pragma unroll
		for ( int h = 0; h < 4; ++h ) {
			float4 s = s4[h * D4 + c];
			o.x += s.x;
			o.y += s.y;
			o.z += s.z;
			o.w += s.w;
		}
		half2* out2 = (half2*)( final_hidden + idx );
		out2[0] = __floats2half2_rn( o.x * 0.25f, o.y * 0.25f );
		out2[1] = __floats2half2_rn( o.z * 0.25f, o.w * 0.25f );
	}
}

} // namespace

void HyperConnection::init_streams(
	const pod::TensorView& in,
	pod::TensorView& streams,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	init_streams( in.ptr<const half>(), streams.ptr<float>(), hidden_size, stream, rows );
}

void HyperConnection::init_streams(
	const half* in,
	float* streams,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	int threads = 256;
	int blocks = ( hidden_size + threads - 1 ) / threads;
	init_streams_kernel<<<dim3( blocks, rows ), dim3( threads ), 0, stream>>>(
		in, streams, hidden_size
	);
}

void HyperConnection::collapse(
	const pod::TensorView& streams,
	const pod::TensorView& fn,
	const pod::TensorView& base,
	const pod::TensorView& scale,
	pod::TensorView& collapsed_out,
	pod::TensorView& post_out,
	pod::TensorView& comb_out,
	pod::TensorView& inv_rms_buf,
	pod::TensorView& mix_logits_buf,
	pod::TensorView& pre_buf,
	int hidden_size,
	int sinkhorn_iters,
	float eps,
	cudaStream_t stream,
	int rows
) {
	collapse(
		streams.ptr<const float>(),
		fn.ptr<const float>(),
		base.ptr<const float>(),
		scale.ptr<const float>(),
		collapsed_out.ptr<half>(),
		post_out.ptr<float>(),
		comb_out.ptr<float>(),
		inv_rms_buf.ptr<float>(),
		mix_logits_buf.ptr<float>(),
		pre_buf.ptr<float>(),
		hidden_size,
		sinkhorn_iters,
		eps,
		stream,
		rows
	);
}

void HyperConnection::collapse(
	const float* streams,
	const float* fn,
	const float* base,
	const float* scale,
	half* collapsed_out,
	float* post_out,
	float* comb_out,
	float* inv_rms_buf,
	float* mix_logits_buf,
	float* pre_buf,
	int hidden_size,
	int sinkhorn_iters,
	float eps,
	cudaStream_t stream,
	int rows
) {
	ENGINE_ASSERT( rows >= 1 && rows <= pod::MAX_ROWS, "HyperConnection::collapse: rows out of range" );
	const int H = 4;
	const int row_len = H * hidden_size;
	const int chunksA = 64;
	const int chunk_cols = ( ( row_len / chunksA + 4 * NUM_THREADS_A - 1 ) / ( 4 * NUM_THREADS_A ) ) * ( 4 * NUM_THREADS_A );
	const int n_chunks_a = ( row_len + chunk_cols - 1 ) / chunk_cols;

	const int chunksC = 4;
	const int chunk_cols_c = ( ( hidden_size / chunksC + 4 * NUM_THREADS - 1 ) / ( 4 * NUM_THREADS ) ) * ( 4 * NUM_THREADS );
	const int n_chunks_c = ( hidden_size + chunk_cols_c - 1 ) / chunk_cols_c;

	dim3 grid_a( n_chunks_a, rows );
	dim3 grid_c( n_chunks_c, rows );

	hc_mix_partials_kernel<4, 24, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(
		streams,
		fn,
		row_len,
		chunk_cols
	);

	hc_mix_finalize_kernel<4, 24, false, true><<<grid_c, NUM_THREADS, 0, stream>>>(
		streams,
		base,
		scale,
		post_out,
		comb_out,
		collapsed_out,
		inv_rms_buf,
		mix_logits_buf,
		pre_buf,
		hidden_size,
		chunksA,
		chunk_cols_c,
		1e-5f,
		eps,
		sinkhorn_iters
	);
}

void HyperConnection::update(
	pod::TensorView& streams,
	const pod::TensorView& sublayer_out,
	const pod::TensorView& post,
	const pod::TensorView& comb,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	update(
		streams.ptr<float>(),
		streams.ptr<const float>(),
		sublayer_out.ptr<const half>(),
		post.ptr<const float>(),
		comb.ptr<const float>(),
		hidden_size,
		stream,
		rows
	);
}

void HyperConnection::update(
	pod::TensorView& streams_out,
	const pod::TensorView& streams_in,
	const pod::TensorView& sublayer_out,
	const pod::TensorView& post,
	const pod::TensorView& comb,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	update(
		streams_out.ptr<float>(),
		streams_in.ptr<const float>(),
		sublayer_out.ptr<const half>(),
		post.ptr<const float>(),
		comb.ptr<const float>(),
		hidden_size,
		stream,
		rows
	);
}

void HyperConnection::update(
	float* streams,
	const half* sublayer_out,
	const float* post,
	const float* comb,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	update( streams, streams, sublayer_out, post, comb, hidden_size, stream, rows );
}

void HyperConnection::update(
	float* streams_out,
	const float* streams_in,
	const half* sublayer_out,
	const float* post,
	const float* comb,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	if ( streams_out != streams_in ) {
		CUDA_CHECK( cudaMemcpyAsync( streams_out, streams_in, static_cast<size_t>( rows ) * 4 * hidden_size * sizeof( float ), cudaMemcpyDeviceToDevice, stream ) );
	}
	int chunks_c = 4;
	int chunk_cols = ( ( hidden_size / chunks_c + 4 * NUM_THREADS - 1 ) / ( 4 * NUM_THREADS ) ) * ( 4 * NUM_THREADS );
	int n_chunks = ( hidden_size + chunk_cols - 1 ) / chunk_cols;
	dim3 grid( n_chunks, rows );
	hc_apply_kernel<4, half, true><<<grid, NUM_THREADS, 0, stream>>>(
		streams_out,
		sublayer_out,
		post,
		comb,
		hidden_size,
		chunk_cols
	);
}

void HyperHead::forward(
	const pod::TensorView& streams,
	pod::TensorView& final_hidden,
	int hidden_size,
	bool is_mean,
	cudaStream_t stream,
	int rows
) {
	if ( is_mean ) {
		mean( streams, final_hidden, hidden_size, stream, rows );
	}
}

void HyperHead::mean(
	const pod::TensorView& streams,
	pod::TensorView& final_hidden,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	mean( streams.ptr<const float>(), final_hidden.ptr<half>(), hidden_size, stream, rows );
}

void HyperHead::mean(
	const float* streams,
	half* final_hidden,
	int hidden_size,
	cudaStream_t stream,
	int rows
) {
	int threads = 256;
	int blocks = ( ( hidden_size / 4 ) + threads - 1 ) / threads;
	mhc_hyperhead_mean_kernel<<<dim3( blocks, rows ), dim3( threads ), 0, stream>>>(
		streams, final_hidden, hidden_size
	);
}

} // namespace engine::modules
