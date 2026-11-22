// Training forms of the qwen4_exp (Flash-Next) layer operations (engine/train/qwen4_exp.h): FP32 norms, the hyper-connection site, the gated add, the causal
// convolutions, the gates of the sublayers and the routing softmax, each forward with its backward.
#include "engine/train/qwen4_exp.h"
#include "engine/config.h"
#include "engine/backend/cuda/rope_math.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>

namespace engine::train::qwen4_exp {

namespace {

#define Q4E_LOOP( i, n ) for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ( n ); i += (size_t) gridDim.x * blockDim.x )

unsigned blocks_for( size_t n ) { return static_cast<unsigned>( ( n + 255 ) / 256 < 65535u * 8u ? ( n + 255 ) / 256 : 65535u * 8u ); }

cudaStream_t cu( backend::stream_t s ) { return static_cast<cudaStream_t>( s ); }

__device__ __forceinline__ float sigmoidf_( float x ) { return 1.0f / ( 1.0f + expf( -x ) ); }

// Sum over a block of 256 threads (all threads get the result)
__device__ __forceinline__ float block_sum256( float v, float* sh ) {
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffffu, v, o );
	const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
	__syncthreads();
	if ( l == 0 ) sh[w] = v;
	__syncthreads();
	float t = 0.0f;
	for ( int i = 0; i < ( blockDim.x >> 5 ); ++i ) t += sh[i];
	return t;
}

__device__ __forceinline__ float weight_at( const float* w, int wrows, bool one_plus, int seg, int n, int i ) {
	if ( w == nullptr ) return 1.0f;
	const float v = w[static_cast<size_t>( seg % wrows ) * n + i];
	return one_plus ? 1.0f + v : v;
}

// One block per segment
__global__ void k_rms_fwd( const float* x, const float* w, int wrows, bool one_plus, bool mean, int n, float eps, float* inv, float* y ) {
	__shared__ float sh[32];
	const int seg = blockIdx.x;
	const float* xs = x + static_cast<size_t>( seg ) * n;
	float s = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) s += xs[i] * xs[i];
	s = block_sum256( s, sh );
	const float iv = rsqrtf( s / ( mean ? static_cast<float>( n ) : 1.0f ) + eps );
	if ( threadIdx.x == 0 ) inv[seg] = iv;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) y[static_cast<size_t>( seg ) * n + i] = xs[i] * iv * weight_at( w, wrows, one_plus, seg, n, i );
}

// dx += inv g - x inv^3 ( g . x ) / N, g = dy * weight
__global__ void k_rms_bwd( const float* x, const float* w, int wrows, bool one_plus, bool mean, int n, const float* inv, const float* dy, float* dx ) {
	__shared__ float sh[32];
	const int seg = blockIdx.x;
	const size_t base = static_cast<size_t>( seg ) * n;
	float s = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) s += dy[base + i] * weight_at( w, wrows, one_plus, seg, n, i ) * x[base + i];
	s = block_sum256( s, sh );
	const float iv = inv[seg];
	const float c = iv * iv * iv * s / ( mean ? static_cast<float>( n ) : 1.0f );
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) dx[base + i] += iv * dy[base + i] * weight_at( w, wrows, one_plus, seg, n, i ) - x[base + i] * c;
}

// A norm's weight gradient: dw[row, i] += sum over the segments of that weight row (seg % wrows == row) of dy x inv (times sigmoid( z ) for the
// gated norm); block (256 columns, weight row, a chunk of its segments), the chunks' sums added atomically
__global__ void k_norm_dw( const float* x, const float* z, const float* inv, const float* dy, int segments, int wrows, int n, int per_chunk, float* dw ) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= n ) return;
	const int row = blockIdx.y;
	const int s0 = row + blockIdx.z * per_chunk * wrows, s1 = min( segments, s0 + per_chunk * wrows );
	float acc = 0.0f;
	for ( int seg = s0; seg < s1; seg += wrows ) {
		const size_t j = static_cast<size_t>( seg ) * n + i;
		const float v = dy[j] * x[j] * inv[seg];
		acc += z ? v * sigmoidf_( z[j] ) : v;
	}
	atomicAdd( &dw[static_cast<size_t>( row ) * n + i], acc );
}

void norm_dw( const float* x, const float* z, const float* inv, const float* dy, int segments, int wrows, int n, float* dw, cudaStream_t st ) {
	const int per_row = ( segments + wrows - 1 ) / wrows, chunks = std::max( 1, std::min( 64, per_row / 64 ) ), per_chunk = ( per_row + chunks - 1 ) / chunks;
	k_norm_dw<<<dim3( ( n + 255 ) / 256, wrows, chunks ), 256, 0, st>>>( x, z, inv, dy, segments, wrows, n, per_chunk, dw );
}

__global__ void k_silu( const float* z, float* t, size_t n ) {
	Q4E_LOOP( i, n ) t[i] = z[i] * sigmoidf_( z[i] );
}

__global__ void k_post( const float* zi, float* post, size_t n ) {
	Q4E_LOOP( i, n ) post[i] = 2.0f * sigmoidf_( zi[i] );
}

// mixed[r, d] = mean_s( sigmoid( u[r, s, d] ) normed[r, s, d] )
__global__ void k_hc_mix( const float* u, const float* normed, int rows, int hidden, int hc, float* mixed ) {
	Q4E_LOOP( i, (size_t) rows * hidden ) {
		const size_t r = i / hidden, d = i % hidden;
		float acc = 0.0f;
		for ( int s = 0; s < hc; ++s ) {
			const size_t j = ( r * hc + s ) * hidden + d;
			acc += sigmoidf_( u[j] ) * normed[j];
		}
		mixed[i] = acc / hc;
	}
}

// dnormed = dmixed sigmoid( u ) / hc, du = dmixed normed sigmoid'( u ) / hc
__global__ void k_hc_mix_bwd( const float* u, const float* normed, const float* dmixed, int rows, int hidden, int hc, float* dnormed, float* du ) {
	Q4E_LOOP( j, (size_t) rows * hc * hidden ) {
		const size_t r = j / ( (size_t) hc * hidden ), d = j % hidden;
		const float g = dmixed[r * hidden + d] / hc, sg = sigmoidf_( u[j] );
		dnormed[j] = g * sg;
		du[j] = g * normed[j] * sg * ( 1.0f - sg );
	}
}

// dz = dt silu'( z ), in place
__global__ void k_silu_bwd( const float* z, float* d, size_t n ) {
	Q4E_LOOP( i, n ) {
		const float s = sigmoidf_( z[i] );
		d[i] *= s * ( 1.0f + z[i] * ( 1.0f - s ) );
	}
}

// dzi = dpost * 2 sigmoid'( zi )
__global__ void k_post_bwd( const float* zi, const float* dpost, float* dzi, size_t n ) {
	Q4E_LOOP( i, n ) {
		const float s = sigmoidf_( zi[i] );
		dzi[i] = dpost[i] * 2.0f * s * ( 1.0f - s );
	}
}

__global__ void k_apply( float* x, const float* post, const float* y, int rows, int hidden, int hc ) {
	Q4E_LOOP( j, (size_t) rows * hc * hidden ) {
		const size_t rs = j / hidden, r = rs / hc, d = j % hidden;
		x[j] += post[rs] * y[r * hidden + d];
	}
}

__global__ void k_apply_bwd_dy( const float* dx, const float* post, int rows, int hidden, int hc, float* dy ) {
	Q4E_LOOP( i, (size_t) rows * hidden ) {
		const size_t r = i / hidden, d = i % hidden;
		float acc = 0.0f;
		for ( int s = 0; s < hc; ++s ) acc += post[r * hc + s] * dx[( r * hc + s ) * hidden + d];
		dy[i] = acc;
	}
}

// One block per (row, stream)
__global__ void k_apply_bwd_dpost( const float* dx, const float* y, int hidden, int hc, float* dpost ) {
	__shared__ float sh[32];
	const int rs = blockIdx.x, r = rs / hc;
	float acc = 0.0f;
	for ( int d = threadIdx.x; d < hidden; d += blockDim.x ) acc += dx[static_cast<size_t>( rs ) * hidden + d] * y[static_cast<size_t>( r ) * hidden + d];
	acc = block_sum256( acc, sh );
	if ( threadIdx.x == 0 ) dpost[rs] = acc;
}

// The causal convolutions see their own sequence only: rows in sequences of L, a tap before its sequence's first row is 0
__global__ void k_conv4_fwd( const float* x, const float* w, int rows, int channels, int dil, int L, float* s, float* y ) {
	Q4E_LOOP( i, (size_t) rows * channels ) {
		const int t = static_cast<int>( i / channels ), c = static_cast<int>( i % channels );
		float acc = 0.0f;
		for ( int j = 0; j < 4; ++j ) {
			const int src = t - ( 3 - j ) * dil;
			if ( t % L >= ( 3 - j ) * dil ) acc += w[static_cast<size_t>( c ) * 4 + j] * x[static_cast<size_t>( src ) * channels + c];
		}
		s[i] = acc;
		y[i] = acc * sigmoidf_( acc );
	}
}

// dx[t, c] += sum_j w[c, j] ds[t + ( 3 - j ) dil, c], ds = dy silu'( s )
__global__ void k_conv4_bwd( const float* s, const float* w, int rows, int channels, int dil, int L, const float* dy, float* dx ) {
	Q4E_LOOP( i, (size_t) rows * channels ) {
		const int t = static_cast<int>( i / channels ), c = static_cast<int>( i % channels );
		float acc = 0.0f;
		for ( int j = 0; j < 4; ++j ) {
			const int dst = t + ( 3 - j ) * dil;
			if ( t % L + ( 3 - j ) * dil < L ) {
				const size_t o = static_cast<size_t>( dst ) * channels + c;
				const float sg = sigmoidf_( s[o] );
				acc += w[static_cast<size_t>( c ) * 4 + j] * dy[o] * sg * ( 1.0f + s[o] * ( 1.0f - sg ) );
			}
		}
		dx[i] += acc;
	}
}

// dw[c, j] += sum_t ds[t, c] x[t - ( 3 - j ) dil, c], ds = dy silu'( s ); x, s, dy rows of stride ld, channels c < n (a thread each), the
// rows in blockIdx.y's span of kConvDwRows (the spans' sums added atomically)
constexpr int kConvDwRows = 64;
__global__ void k_conv4_dw( const float* x, const float* s, const float* dy, int rows, int n, int ld, int dil, int L, float* dw ) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= n ) return;
	float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	const int t1 = min( rows, static_cast<int>( blockIdx.y + 1 ) * kConvDwRows );
	for ( int t = blockIdx.y * kConvDwRows; t < t1; ++t ) {
		const size_t o = static_cast<size_t>( t ) * ld + c;
		const float sg = sigmoidf_( s[o] ), ds = dy[o] * sg * ( 1.0f + s[o] * ( 1.0f - sg ) );
		for ( int j = 0; j < 4; ++j ) {
			const int src = t - ( 3 - j ) * dil;
			if ( t % L >= ( 3 - j ) * dil ) acc[j] += ds * x[static_cast<size_t>( src ) * ld + c];
		}
	}
	for ( int j = 0; j < 4; ++j ) atomicAdd( &dw[static_cast<size_t>( c ) * 4 + j], acc[j] );
}

__global__ void k_sig_gate( const float* a, const float* g, float* o, size_t n ) {
	Q4E_LOOP( i, n ) o[i] = a[i] * sigmoidf_( g[i] );
}

__global__ void k_sig_gate_bwd( const float* a, const float* g, const float* dout, float* da, float* dg, size_t n ) {
	Q4E_LOOP( i, n ) {
		const float s = sigmoidf_( g[i] );
		if ( da ) da[i] += dout[i] * s;
		if ( dg ) dg[i] += dout[i] * a[i] * s * ( 1.0f - s );
	}
}

__global__ void k_grms_fwd( const float* x, const float* z, const float* gamma, int n, float eps, float* inv, float* y ) {
	__shared__ float sh[32];
	const int seg = blockIdx.x;
	const size_t base = static_cast<size_t>( seg ) * n;
	float s = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) s += x[base + i] * x[base + i];
	s = block_sum256( s, sh );
	const float iv = rsqrtf( s / n + eps );
	if ( threadIdx.x == 0 ) inv[seg] = iv;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) y[base + i] = x[base + i] * iv * gamma[i] * sigmoidf_( z[base + i] );
}

__global__ void k_grms_bwd( const float* x, const float* z, const float* gamma, int n, const float* inv, const float* dy, float* dx, float* dz ) {
	__shared__ float sh[32];
	const int seg = blockIdx.x;
	const size_t base = static_cast<size_t>( seg ) * n;
	const float iv = inv[seg];
	float s = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) s += dy[base + i] * gamma[i] * sigmoidf_( z[base + i] ) * x[base + i];
	s = block_sum256( s, sh );
	const float c = iv * iv * iv * s / n;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
		const float sg = sigmoidf_( z[base + i] );
		dx[base + i] += iv * dy[base + i] * gamma[i] * sg - x[base + i] * c;
		dz[base + i] += dy[base + i] * x[base + i] * iv * gamma[i] * sg * ( 1.0f - sg );
	}
}

__global__ void k_slots_to_dense( const float* v, const int* ids, int rows, int k, int experts, float* out ) {
	Q4E_LOOP( i, (size_t) rows * k ) {
		const int r = static_cast<int>( i / k ), e = ids[i];
		if ( e >= 0 && e < experts ) out[static_cast<size_t>( r ) * experts + e] += v[i];
	}
}

__global__ void k_topk_softmax_bwd( const float* w, const float* dw, int rows, int k, float* dlogit ) {
	Q4E_LOOP( r, (size_t) rows ) {
		float s = 0.0f;
		for ( int j = 0; j < k; ++j ) s += w[r * k + j] * dw[r * k + j];
		for ( int j = 0; j < k; ++j ) dlogit[r * k + j] = w[r * k + j] * ( dw[r * k + j] - s );
	}
}

// ---- GDN gated delta rule ----
constexpr int GD = 128;   // head dim (keys and values)
constexpr int GW = 16;    // value columns per block, a warp each

// The forward's row tiles: kGdnTile rows of the key head's k and q, the value head's e^gate, beta and k . q, and the block's 16 value columns,
// staged in shared memory, the next tile's loads in flight while a tile is used
constexpr int kGdnTile = 16;

// Block (head, 16 columns, sequence) of 8 warps, a half-warp per column holding its state row in 8 values per lane (element r * 16 + l);
// the rows hold gridDim.z sequences of rows / gridDim.z, each starting from a zero state. o_t[c] = scale S_c . q_t after the update is
// e^gate (S_c . q_t) + d (k_t . q_t) from the state before it, so both sums of a row run together
__global__ __launch_bounds__( 256 )
void k_gdn_fwd( const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ gate,
	const float* __restrict__ beta, int rows, int kh, int vh, float scale, int chunk, float* __restrict__ ckpt, float* __restrict__ o ) {
	__shared__ float4 sk4[kGdnTile * GD / 4], sq4[kGdnTile * GD / 4];
	__shared__ float sg[kGdnTile], sb[kGdnTile], skq[kGdnTile], sv[kGdnTile][GW];
	const int h = blockIdx.x, c0 = blockIdx.y * GW, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l = lane & 15, khh = h / ( vh / kh );
	const int cl = w * 2 + ( lane >> 4 ), c = c0 + cl;
	const int L = rows / gridDim.z, nch = ( L + chunk - 1 ) / chunk, s0 = blockIdx.z * L, s1 = s0 + L;
	const float* sk = reinterpret_cast<const float*>( sk4 );
	const float* sq = reinterpret_cast<const float*>( sq4 );
	// this thread's share of a tile: row tid / 16 of k and q (8 values each: two float4 at part tid % 16), one v value, gate or beta
	const int lr = tid >> 4, lp = tid & 15;
	float4 pk[2], pq[2];
	float pv = 0.0f, pgb = 0.0f;
	auto fetch = [&]( int t0 ) {
		const int t = t0 + lr;
		if ( t < s1 ) {
			const float4* kt = reinterpret_cast<const float4*>( k + ( static_cast<size_t>( t ) * kh + khh ) * GD ) + lp * 2;
			const float4* qt = reinterpret_cast<const float4*>( q + ( static_cast<size_t>( t ) * kh + khh ) * GD ) + lp * 2;
			pk[0] = kt[0]; pk[1] = kt[1]; pq[0] = qt[0]; pq[1] = qt[1];
			pv = v[( static_cast<size_t>( t ) * vh + h ) * GD + c0 + lp];
		}
		else {
			pk[0] = pk[1] = pq[0] = pq[1] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
			pv = 0.0f;
		}
		const int tg = t0 + ( tid & 15 );
		if ( tid < 32 ) pgb = tg >= s1 ? 0.0f : tid < 16 ? expf( gate[static_cast<size_t>( tg ) * vh + h] ) : beta[static_cast<size_t>( tg ) * vh + h];
	};
	float S[8];
	for ( int r = 0; r < 8; ++r ) S[r] = 0.0f;
	fetch( s0 );
	for ( int t0 = s0; t0 < s1; t0 += kGdnTile ) {
		__syncthreads();   // the previous tile used
		sk4[lr * ( GD / 4 ) + lp * 2] = pk[0]; sk4[lr * ( GD / 4 ) + lp * 2 + 1] = pk[1];
		sq4[lr * ( GD / 4 ) + lp * 2] = pq[0]; sq4[lr * ( GD / 4 ) + lp * 2 + 1] = pq[1];
		sv[lr][lp] = pv;
		if ( tid < 16 ) sg[tid] = pgb;
		else if ( tid < 32 ) sb[tid - 16] = pgb;
		__syncthreads();
		{
			// k . q of the tile's rows: rows w * 2 + half-warp
			const int i = w * 2 + ( lane >> 4 );
			float kq = 0.0f;
			for ( int r = 0; r < 8; ++r ) kq = fmaf( sk[i * GD + r * 16 + l], sq[i * GD + r * 16 + l], kq );
			for ( int m = 8; m > 0; m >>= 1 ) kq += __shfl_xor_sync( 0xffffffffu, kq, m );
			if ( l == 0 ) skq[i] = kq;
		}
		if ( t0 + kGdnTile < s1 ) fetch( t0 + kGdnTile );
		__syncthreads();
		const int n = min( kGdnTile, s1 - t0 );
		for ( int i = 0; i < n; ++i ) {
			const int t = t0 + i;
			if ( ( t - s0 ) % chunk == 0 ) {
				float* ck = ckpt + ( ( ( static_cast<size_t>( blockIdx.z ) * nch + ( t - s0 ) / chunk ) * vh + h ) * GD + c ) * GD;
				for ( int r = 0; r < 8; ++r ) ck[r * 16 + l] = S[r];
			}
			float kk[8], u = 0.0f, sqv = 0.0f;
			for ( int r = 0; r < 8; ++r ) {
				kk[r] = sk[i * GD + r * 16 + l];
				u = fmaf( S[r], kk[r], u );
				sqv = fmaf( S[r], sq[i * GD + r * 16 + l], sqv );
			}
			for ( int m = 8; m > 0; m >>= 1 ) {
				u += __shfl_xor_sync( 0xffffffffu, u, m );
				sqv += __shfl_xor_sync( 0xffffffffu, sqv, m );
			}
			const float g = sg[i], d = sb[i] * ( sv[i][cl] - g * u );
			for ( int r = 0; r < 8; ++r ) S[r] = fmaf( g, S[r], d * kk[r] );
			if ( l == 0 ) o[( static_cast<size_t>( t ) * vh + h ) * GD + c] = ( g * sqv + d * skq[i] ) * scale;
		}
	}
}

// The backward's sub-chunks: kGdnSub rows whose states a thread holds in registers at once
constexpr int kGdnSub = 8;

// The backward, block (head, 16 columns, sequence) of 8 warps in the forward's layout, the chunks last first and dS in registers throughout.
// Per chunk: from its checkpoint the states at its sub-chunks' starts (into sub, a column's kGdnSub-row sub-checkpoints); then per
// sub-chunk, last first, its states recomputed into registers and its rows in reverse. A row's sums over the block's columns (dq, dk,
// dbeta, dgate) go through per-warp partials in shared memory (two buffers, one barrier a row; shared float atomics are compare-and-swap
// loops on SM70) and one global add per entry
__global__ __launch_bounds__( 256, 2 )
void k_gdn_bwd( const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ gate,
	const float* __restrict__ beta, int rows, int kh, int vh, float scale, int chunk, const float* __restrict__ ckpt, const float* __restrict__ dout,
	float* dq, float* dk, float* dv, float* dgate, float* dbeta, float* __restrict__ sub ) {
	__shared__ float4 sk4[kGdnSub * GD / 4], sq4[kGdnSub * GD / 4];
	__shared__ float sg[kGdnSub], sb[kGdnSub], sv[kGdnSub][GW], sdo[kGdnSub][GW], su[kGdnSub][GW], sd[kGdnSub][GW];
	__shared__ float sp[2][8][2 * GD], sc[2][2][GW];   // per warp a row's dq | dk partials; per column its dbeta, dgate terms
	const int h = blockIdx.x, c0 = blockIdx.y * GW, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l = lane & 15, khh = h / ( vh / kh );
	const int cl = w * 2 + ( lane >> 4 ), c = c0 + cl;
	const int L = rows / gridDim.z, nch = ( L + chunk - 1 ) / chunk, s0 = blockIdx.z * L, s1 = s0 + L, nsub = ( chunk + kGdnSub - 1 ) / kGdnSub;
	const float* skf = reinterpret_cast<const float*>( sk4 );
	const float* sqf = reinterpret_cast<const float*>( sq4 );
	float* subc = sub + ( ( static_cast<size_t>( blockIdx.z ) * vh + h ) * GD + c ) * nsub * GD;
	int buf = 0;
	// rows [t0, t0 + n) into the tiles: k, e^gate, beta, v, with all also q and dout
	auto stage = [&]( int t0, int n, bool all ) {
		__syncthreads();
		const int r = tid >> 5, p = tid & 31;
		const size_t kb = ( static_cast<size_t>( t0 + r ) * kh + khh ) * GD;
		sk4[tid] = r < n ? reinterpret_cast<const float4*>( k + kb )[p] : make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
		if ( all ) sq4[tid] = r < n ? reinterpret_cast<const float4*>( q + kb )[p] : make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
		const int rv = ( tid & 127 ) >> 4, j = tid & 15;
		const size_t vb = ( static_cast<size_t>( t0 + rv ) * vh + h ) * GD + c0 + j;
		if ( tid < 128 ) sv[rv][j] = rv < n ? v[vb] : 0.0f;
		else if ( all ) sdo[rv][j] = rv < n ? dout[vb] : 0.0f;
		if ( tid < kGdnSub ) sg[tid] = tid < n ? expf( gate[static_cast<size_t>( t0 + tid ) * vh + h] ) : 0.0f;
		else if ( tid < 2 * kGdnSub ) sb[tid - kGdnSub] = tid - kGdnSub < n ? beta[static_cast<size_t>( t0 + tid - kGdnSub ) * vh + h] : 0.0f;
		__syncthreads();
	};
	float dS[8];
	for ( int r = 0; r < 8; ++r ) dS[r] = 0.0f;
	for ( int n = nch - 1; n >= 0; --n ) {
		const int cs = s0 + n * chunk, ce = min( s1, cs + chunk ), ns = ( ce - cs + kGdnSub - 1 ) / kGdnSub;
		const float* ck = ckpt + ( ( ( static_cast<size_t>( blockIdx.z ) * nch + n ) * vh + h ) * GD + c ) * GD;
		float S[8];
		for ( int r = 0; r < 8; ++r ) S[r] = ck[r * 16 + l];
		// the states at the starts of sub-chunks 1 .. ns - 1 (the last one stays in S)
		for ( int j = 0; j + 1 < ns; ++j ) {
			stage( cs + j * kGdnSub, kGdnSub, false );
			for ( int i = 0; i < kGdnSub; ++i ) {
				float kk[8], u = 0.0f;
				for ( int r = 0; r < 8; ++r ) {
					kk[r] = skf[i * GD + r * 16 + l];
					u = fmaf( S[r], kk[r], u );
				}
				for ( int m = 8; m > 0; m >>= 1 ) u += __shfl_xor_sync( 0xffffffffu, u, m );
				const float g = sg[i], d = sb[i] * ( sv[i][cl] - g * u );
				for ( int r = 0; r < 8; ++r ) S[r] = fmaf( g, S[r], d * kk[r] );
			}
			if ( j + 2 < ns ) for ( int r = 0; r < 8; ++r ) subc[( j + 1 ) * GD + r * 16 + l] = S[r];
		}
		for ( int j = ns - 1; j >= 0; --j ) {
			const int t0 = cs + j * kGdnSub, nr = min( kGdnSub, ce - t0 );
			if ( j < ns - 1 ) {
				const float* from = j == 0 ? ck : subc + j * GD;
				for ( int r = 0; r < 8; ++r ) S[r] = from[r * 16 + l];
			}
			stage( t0, nr, true );
			// the sub-chunk's states before each row (Sx), u and d per row and column
			float Sx[kGdnSub][8];
#pragma unroll
			for ( int i = 0; i < kGdnSub; ++i ) {
				for ( int r = 0; r < 8; ++r ) Sx[i][r] = S[r];
				if ( i < nr ) {
					float kk[8], u = 0.0f;
					for ( int r = 0; r < 8; ++r ) {
						kk[r] = skf[i * GD + r * 16 + l];
						u = fmaf( S[r], kk[r], u );
					}
					for ( int m = 8; m > 0; m >>= 1 ) u += __shfl_xor_sync( 0xffffffffu, u, m );
					const float g = sg[i], d = sb[i] * ( sv[i][cl] - g * u );
					if ( l == 0 ) {
						su[i][cl] = u;
						sd[i][cl] = d;
					}
					for ( int r = 0; r < 8; ++r ) S[r] = fmaf( g, S[r], d * kk[r] );
				}
			}
			__syncwarp();
#pragma unroll
			for ( int i = kGdnSub - 1; i >= 0; --i ) {
				if ( i >= nr ) continue;
				const int t = t0 + i;
				const float g = sg[i], bt = sb[i], u = su[i][cl], d = sd[i][cl], sdoc = scale * sdo[i][cl];
				float kk[8], dqc[8], dd = 0.0f, dgp = 0.0f;
				for ( int r = 0; r < 8; ++r ) {
					kk[r] = skf[i * GD + r * 16 + l];
					dqc[r] = sdoc * fmaf( g, Sx[i][r], d * kk[r] );   // scale doc S_t
					dS[r] = fmaf( sdoc, sqf[i * GD + r * 16 + l], dS[r] );
					dd = fmaf( dS[r], kk[r], dd );
					dgp = fmaf( dS[r], Sx[i][r], dgp );
				}
				for ( int m = 8; m > 0; m >>= 1 ) {
					dd += __shfl_xor_sync( 0xffffffffu, dd, m );
					dgp += __shfl_xor_sync( 0xffffffffu, dgp, m );
				}
				const float du = -bt * g * dd;
				for ( int r = 0; r < 8; ++r ) {
					float dkc = fmaf( dS[r], d, Sx[i][r] * du );
					dS[r] = fmaf( g, dS[r], du * kk[r] );
					const float a = dqc[r] + __shfl_xor_sync( 0xffffffffu, dqc[r], 16 );
					dkc += __shfl_xor_sync( 0xffffffffu, dkc, 16 );
					if ( lane < 16 ) {
						sp[buf][w][r * 16 + l] = a;
						sp[buf][w][GD + r * 16 + l] = dkc;
					}
				}
				if ( l == 0 ) {
					dv[( static_cast<size_t>( t ) * vh + h ) * GD + c] += bt * dd;
					sc[buf][0][cl] = dd * ( sv[i][cl] - g * u );
					sc[buf][1][cl] = ( dgp - bt * dd * u ) * g;
				}
				__syncthreads();
				float s = 0.0f;
				for ( int x = 0; x < 8; ++x ) s += sp[buf][x][tid];
				atomicAdd( ( tid < GD ? dq : dk ) + ( static_cast<size_t>( t ) * kh + khh ) * GD + ( tid & ( GD - 1 ) ), s );
				if ( tid < 2 ) {
					float z = 0.0f;
					for ( int x = 0; x < GW; ++x ) z += sc[buf][tid][x];
					atomicAdd( ( tid == 0 ? dbeta : dgate ) + static_cast<size_t>( t ) * vh + h, z );
				}
				buf ^= 1;
			}
		}
	}
}

__global__ void k_gdn_gates( const float* a, const float* b, const float* A_log, const float* dt_bias, int rows, int vh, float* gate, float* beta ) {
	Q4E_LOOP( i, (size_t) rows * vh ) {
		const int h = static_cast<int>( i % vh );
		const float x = a[i] + dt_bias[h];
		const float sp = x > 20.0f ? x : log1pf( expf( x ) );
		gate[i] = -expf( A_log[h] ) * sp;
		beta[i] = sigmoidf_( b[i] );
	}
}

__global__ void k_gdn_gates_bwd( const float* a, const float* b, const float* A_log, const float* dt_bias, int rows, int vh, const float* dgate,
	const float* dbeta, float* da, float* db ) {
	Q4E_LOOP( i, (size_t) rows * vh ) {
		const int h = static_cast<int>( i % vh );
		const float x = a[i] + dt_bias[h];
		da[i] = dgate[i] * -expf( A_log[h] ) * ( x > 20.0f ? 1.0f : sigmoidf_( x ) );
		const float s = sigmoidf_( b[i] );
		db[i] = dbeta[i] * s * ( 1.0f - s );
	}
}

// ---- attention ----
// One thread per (row, head, pair)
__global__ void k_rope( float* x, int rows, int heads, int head_dim, int rot_dim, float base, float sign, int seq_len ) {
	const int pairs = rot_dim / 2;
	Q4E_LOOP( i, (size_t) rows * heads * pairs ) {
		const int p = static_cast<int>( i % pairs );
		const size_t rh = i / pairs;
		const int r = static_cast<int>( rh / heads ) % seq_len;   // the position in its sequence
		float c, s;
		engine::math::rope_cos_sin<true>( r, p, rot_dim, base, c, s );
		s *= sign;
		float* xh = x + rh * head_dim;
		const float a = xh[p], b = xh[p + pairs];
		xh[p] = a * c - b * s;
		xh[p + pairs] = a * s + b * c;
	}
}

__global__ void k_unpack_qg( const float* qg, int rows, int heads, int hd, float* q, float* g ) {
	Q4E_LOOP( i, (size_t) rows * heads * hd ) {
		const size_t rh = i / hd, d = i % hd;
		q[i] = qg[rh * 2 * hd + d];
		g[i] = qg[rh * 2 * hd + hd + d];
	}
}

__global__ void k_pack_qg( const float* dq, const float* dg, int rows, int heads, int hd, float* dqg ) {
	Q4E_LOOP( i, (size_t) rows * heads * hd ) {
		const size_t rh = i / hd, d = i % hd;
		dqg[rh * 2 * hd + d] = dq[i];
		dqg[rh * 2 * hd + hd + d] = dg[i];
	}
}

// One block per row i of a rows x rows score matrix: softmax over j <= i, zero above
// The keys a query at position p sees: causal, and past the QSA budget (more than budget complete pools of kQsaPool tokens) only its
// selected pools (bits: a row of words per position) and its incomplete tail
__device__ __forceinline__ bool qsa_sees( int p, int j, const uint32_t* row_bits, int budget ) {
	if ( j > p ) return false;
	const int vis = p + 1;
	if ( row_bits == nullptr || vis / kQsaPool <= budget || j >= vis - vis % kQsaPool ) return true;
	const int pool = j / kQsaPool;
	return ( row_bits[pool >> 5] >> ( pool & 31 ) ) & 1u;
}

// Softmax per row of a query block (rows r0 .. r0 + gridDim.x of the sequence, ld keys), over the keys it sees (the others 0)
__global__ void k_attn_softmax( float* s, int ld, int r0, const uint32_t* bits, int words, int budget ) {
	__shared__ float sh[32];
	const int i = blockIdx.x, p = r0 + i;
	float* row = s + static_cast<size_t>( i ) * ld;
	const uint32_t* rb = bits ? bits + static_cast<size_t>( p ) * words : nullptr;
	float m = -INFINITY;
	for ( int j = threadIdx.x; j <= p; j += blockDim.x ) if ( qsa_sees( p, j, rb, budget ) ) m = fmaxf( m, row[j] );
	for ( int o = 16; o > 0; o >>= 1 ) m = fmaxf( m, __shfl_xor_sync( 0xffffffffu, m, o ) );
	__syncthreads();
	if ( ( threadIdx.x & 31 ) == 0 ) sh[threadIdx.x >> 5] = m;
	__syncthreads();
	m = -INFINITY;
	for ( int w = 0; w < ( blockDim.x >> 5 ); ++w ) m = fmaxf( m, sh[w] );
	float sum = 0.0f;
	for ( int j = threadIdx.x; j <= p; j += blockDim.x ) if ( qsa_sees( p, j, rb, budget ) ) sum += expf( row[j] - m );
	sum = block_sum256( sum, sh );
	const float inv = 1.0f / sum;
	for ( int j = threadIdx.x; j < ld; j += blockDim.x ) row[j] = qsa_sees( p, j, rb, budget ) ? expf( row[j] - m ) * inv : 0.0f;
}

// dp <- p ( dp - sum_j p dp ), per row of ld (p is 0 where a row sees nothing)
__global__ void k_softmax_bwd( const float* p, float* dp, int ld ) {
	__shared__ float sh[32];
	const int i = blockIdx.x;
	const float* pr = p + static_cast<size_t>( i ) * ld;
	float* dr = dp + static_cast<size_t>( i ) * ld;
	float s = 0.0f;
	for ( int j = threadIdx.x; j < ld; j += blockDim.x ) s += pr[j] * dr[j];
	s = block_sum256( s, sh );
	for ( int j = threadIdx.x; j < ld; j += blockDim.x ) dr[j] = pr[j] * ( dr[j] - s );
}

// ---- QSA: Flash-Next's selection of key pools past its budget (engine/modules/qsa.h), held fixed like the routing (no gradient) ----
__device__ __forceinline__ float h16( float x ) { return __half2float( __float2half_rn( x ) ); }

// Sum over a block of 128 threads (all get it)
__device__ __forceinline__ float block_sum128_( float v, float* sh ) {
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffffu, v, o );
	__syncthreads();
	if ( ( threadIdx.x & 31 ) == 0 ) sh[threadIdx.x >> 5] = v;
	__syncthreads();
	return sh[0] + sh[1] + sh[2] + sh[3];
}

__device__ __forceinline__ float qsa_rope( float x, float partner, int d, int pos, int rot_dim, float theta ) {
	const int half_rot = rot_dim / 2;
	if ( d >= rot_dim ) return x;
	float c, s;
	engine::math::rope_cos_sin<false>( pos, d % half_rot, rot_dim, theta, c, s );
	return d < half_rot ? x * c - partner * s : x * c + partner * s;
}

// Block (row, head) of 128 threads, the inference kernel's roundings: the projection's output in FP16, q = rmsnorm * (1 + w) in FP16, roped at
// the row's position in its sequence; head kQsaHeads: the raw key (FP16) into raw_k [rows, 128]
__global__ __launch_bounds__( 128 )
void k_qsa_q( const float* qk, const __half* qw, int seq_len, int rot_dim, float theta, float eps, float* q, float* raw_k ) {
	__shared__ float xs[kQsaDim], sh[4];
	const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x, pos = r % seq_len;
	const float x = h16( qk[static_cast<size_t>( r ) * kQsaProj + h * kQsaDim + d] );
	if ( h == kQsaHeads ) {
		raw_k[static_cast<size_t>( r ) * kQsaDim + d] = x;
		return;
	}
	const float ms = block_sum128_( x * x, sh ) / kQsaDim;
	const float xn = h16( x * rsqrtf( ms + eps ) * ( 1.0f + __half2float( qw[d] ) ) );
	xs[d] = xn;
	__syncthreads();
	const int half_rot = rot_dim / 2;
	const float partner = d < rot_dim ? xs[d < half_rot ? d + half_rot : d - half_rot] : 0.0f;
	q[( static_cast<size_t>( r ) * kQsaHeads + h ) * kQsaDim + d] = h16( qsa_rope( xn, partner, d, pos, rot_dim, theta ) );
}

// Block (pool, sequence): the pool's key = rope at its first position( rmsnorm( mean of its raw keys, FP16 ) * (1 + w) ), FP16
__global__ __launch_bounds__( 128 )
void k_qsa_pool( const float* raw_k, const __half* kw, int seq_len, int rot_dim, float theta, float eps, float* pools ) {
	__shared__ float xs[kQsaDim], sh[4];
	const int p = blockIdx.x, s = blockIdx.y, d = threadIdx.x, np = seq_len / kQsaPool;
	const float* rk = raw_k + ( static_cast<size_t>( s ) * seq_len + static_cast<size_t>( p ) * kQsaPool ) * kQsaDim;
	float acc = 0.0f;
	for ( int j = 0; j < kQsaPool; ++j ) acc += rk[j * kQsaDim + d];
	const float mean = h16( acc / kQsaPool );
	const float ms = block_sum128_( mean * mean, sh ) / kQsaDim;
	const float xn = h16( mean * rsqrtf( ms + eps ) * ( 1.0f + __half2float( kw[d] ) ) );
	xs[d] = xn;
	__syncthreads();
	const int half_rot = rot_dim / 2;
	const float partner = d < rot_dim ? xs[d < half_rot ? d + half_rot : d - half_rot] : 0.0f;
	pools[( static_cast<size_t>( s ) * np + p ) * kQsaDim + d] = h16( qsa_rope( xn, partner, d, p * kQsaPool, rot_dim, theta ) );
}

// Grid (pool tiles of 128, query rows r0 ..): score[i][p] = scale sum_h relu( q[h] . key[p] ) (FP16) for the pools p < (pos + 1) / kQsaPool of the
// rows past the budget
__global__ __launch_bounds__( 128 )
void k_qsa_score( const float* q, const float* pools, int r0, int np, int budget, float scale, float* scores ) {
	__shared__ float qs[kQsaHeads * kQsaDim];
	const int i = blockIdx.y, pos = r0 + i, bound = ( pos + 1 ) / kQsaPool;
	if ( bound <= budget ) return;
	for ( int x = threadIdx.x; x < kQsaHeads * kQsaDim; x += 128 ) qs[x] = q[static_cast<size_t>( pos ) * kQsaHeads * kQsaDim + x];
	__syncthreads();
	const int p = blockIdx.x * 128 + threadIdx.x;
	if ( p >= bound ) return;
	const float4* key = reinterpret_cast<const float4*>( pools + static_cast<size_t>( p ) * kQsaDim );
	float s[kQsaHeads] = { 0.0f, 0.0f, 0.0f, 0.0f };
	for ( int d4 = 0; d4 < kQsaDim / 4; ++d4 ) {
		const float4 kv = key[d4];
		for ( int h = 0; h < kQsaHeads; ++h ) {
			const float* qh = qs + h * kQsaDim + d4 * 4;
			s[h] = fmaf( qh[0], kv.x, fmaf( qh[1], kv.y, fmaf( qh[2], kv.z, fmaf( qh[3], kv.w, s[h] ) ) ) );
		}
	}
	float total = 0.0f;
	for ( int h = 0; h < kQsaHeads; ++h ) total += fmaxf( s[h], 0.0f );
	scores[static_cast<size_t>( i ) * np + p] = h16( total * scale );
}

__device__ __forceinline__ unsigned qsa_key( float s ) {
	const unsigned b = __half_as_ushort( __float2half_rn( s ) );
	return ( b & 0x8000u ) ? ( ~b & 0xFFFFu ) : ( b | 0x8000u );
}

// Block per query row: the budget best pools by FP16 score (ties at the threshold in pool order, as DSA::select) as bits of the row's words;
// rows within the budget get no bits (they see every key)
__global__ __launch_bounds__( 1024 )
void k_qsa_select( const float* scores, int np, int r0, int budget, int words, uint32_t* bits ) {
	__shared__ int hist[256];
	__shared__ uint32_t s_bits[kQsaMaxWords];
	__shared__ int s_b1, s_above1, s_thr, s_need_eq, s_wsum[32], s_carry_eq;
	const int i = blockIdx.x, t = threadIdx.x, pos = r0 + i, bound = ( pos + 1 ) / kQsaPool;
	uint32_t* out = bits + static_cast<size_t>( pos ) * words;
	for ( int w = t; w < words; w += 1024 ) s_bits[w] = 0u;
	if ( bound <= budget ) {
		for ( int w = t; w < words; w += 1024 ) out[w] = 0u;
		return;
	}
	const float* sc = scores + static_cast<size_t>( i ) * np;
	if ( t < 256 ) hist[t] = 0;
	__syncthreads();
	for ( int p = t; p < bound; p += 1024 ) atomicAdd( &hist[qsa_key( sc[p] ) >> 8], 1 );
	__syncthreads();
	if ( t == 0 ) {
		int cum = 0, b = 255;
		for ( ; b > 0; --b ) {
			if ( cum + hist[b] >= budget ) break;
			cum += hist[b];
		}
		s_b1 = b;
		s_above1 = cum;
	}
	__syncthreads();
	if ( t < 256 ) hist[t] = 0;
	__syncthreads();
	for ( int p = t; p < bound; p += 1024 ) {
		const unsigned key = qsa_key( sc[p] );
		if ( ( key >> 8 ) == static_cast<unsigned>( s_b1 ) ) atomicAdd( &hist[key & 255u], 1 );
	}
	__syncthreads();
	if ( t == 0 ) {
		const int rem = budget - s_above1;
		int cum = 0, b = 255;
		for ( ; b > 0; --b ) {
			if ( cum + hist[b] >= rem ) break;
			cum += hist[b];
		}
		s_thr = ( s_b1 << 8 ) | b;
		s_need_eq = rem - cum;
		s_carry_eq = 0;
	}
	__syncthreads();
	const unsigned thr = static_cast<unsigned>( s_thr );
	const int need_eq = s_need_eq;
	// strictly better pools all; the ties at the threshold first in pool order
	for ( int base = 0; base < bound; base += 1024 ) {
		const int p = base + t;
		const unsigned key = p < bound ? qsa_key( sc[p] ) : 0u;
		const bool gt = p < bound && key > thr, eq = p < bound && key == thr;
		const unsigned be = __ballot_sync( 0xffffffffu, eq );
		if ( ( t & 31 ) == 0 ) s_wsum[t >> 5] = __popc( be );
		__syncthreads();
		int we = 0;
		for ( int w2 = 0; w2 < ( t >> 5 ); ++w2 ) we += s_wsum[w2];
		const int pos_eq = s_carry_eq + we + __popc( be & ( ( 1u << ( t & 31 ) ) - 1u ) );
		if ( gt || ( eq && pos_eq < need_eq ) ) atomicOr( &s_bits[p >> 5], 1u << ( p & 31 ) );
		__syncthreads();
		if ( t == 0 ) for ( int w2 = 0; w2 < 32; ++w2 ) s_carry_eq += s_wsum[w2];
		__syncthreads();
	}
	for ( int w = t; w < words; w += 1024 ) out[w] = s_bits[w];
}

// ---- routing ----
// One block per row, a thread per expert: k rounds of a block argmax (ties to the lowest index)
__global__ void k_router_topk( const float* logits, int experts, int k, int* ids, float* w ) {
	__shared__ float s_v[32];
	__shared__ int s_i[32];
	__shared__ float s_sel[16];
	const int r = blockIdx.x, e = threadIdx.x;
	float my = e < experts ? logits[static_cast<size_t>( r ) * experts + e] : -INFINITY;
	bool taken = e >= experts;
	for ( int j = 0; j < k; ++j ) {
		float v = taken ? -INFINITY : my;
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
			s_sel[j] = bv;
			s_i[0] = bi;
		}
		__syncthreads();
		if ( e == s_i[0] ) taken = true;
		__syncthreads();
	}
	if ( e == 0 ) {
		float sum = 0.0f;
		for ( int j = 0; j < k; ++j ) sum += expf( s_sel[j] - s_sel[0] );
		for ( int j = 0; j < k; ++j ) w[static_cast<size_t>( r ) * k + j] = expf( s_sel[j] - s_sel[0] ) / sum;
	}
}

// Block per row, a thread per expert: the row's slots (its k selected experts, then the c best of the others, ties to the lowest index) with
// their weights (the selected ones' and 0), and the weight each candidate would take in the softmax over the selected ones and itself
__global__ void k_route_candidates( const float* logits, const int* ids, const float* w, int experts, int k, int c, int* sids, float* sw, float* wp ) {
	__shared__ float s_v[32];
	__shared__ int s_i[32];
	__shared__ int s_taken[1024];
	const int r = blockIdx.x, e = threadIdx.x, ks = k + c;
	const float* lr = logits + static_cast<size_t>( r ) * experts;
	if ( e < experts ) s_taken[e] = 0;
	__syncthreads();
	if ( e < k ) s_taken[ids[static_cast<size_t>( r ) * k + e]] = 1;
	__syncthreads();
	const float my = e < experts ? lr[e] : -INFINITY;
	for ( int j = 0; j < c; ++j ) {
		const bool taken = e >= experts || s_taken[e];
		float v = taken ? -INFINITY : my;
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
			sids[static_cast<size_t>( r ) * ks + k + j] = bi;
			s_taken[bi] = 1;
		}
		__syncthreads();
	}
	if ( e == 0 ) {
		float mx = -INFINITY;
		for ( int j = 0; j < k; ++j ) mx = fmaxf( mx, lr[ids[static_cast<size_t>( r ) * k + j]] );
		float sum = 0.0f;
		for ( int j = 0; j < k; ++j ) sum += expf( lr[ids[static_cast<size_t>( r ) * k + j]] - mx );
		for ( int j = 0; j < k; ++j ) {
			sids[static_cast<size_t>( r ) * ks + j] = ids[static_cast<size_t>( r ) * k + j];
			sw[static_cast<size_t>( r ) * ks + j] = w[static_cast<size_t>( r ) * k + j];
		}
		for ( int j = 0; j < c; ++j ) {
			const float ec = expf( lr[sids[static_cast<size_t>( r ) * ks + k + j]] - mx );
			sw[static_cast<size_t>( r ) * ks + k + j] = 0.0f;
			wp[static_cast<size_t>( r ) * c + j] = ec / ( sum + ec );
		}
	}
}

// Routing margins per row, a warp each (lane 0 the pairs): A the teacher's top k (its first k ids), B the model's selection; every wrong pick w
// (in B, not in A) against every miss m (in A, not in B): softplus( l_w - l_m + delta ); the weakest agreed pick c against the strongest expert
// neither picks d: softplus( delta - ( l_c - l_d ) ). loss[r] = the row's sum; dl (rows x experts, zeroed) receives its gradient
__global__ void k_route_margin( const float* logits, const int* sel, const int* tid, int rows, int experts, int k, int tk, float delta, float* dl,
	float* loss ) {
	const int row = static_cast<int>( ( static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x ) / 32 ), lane = threadIdx.x % 32;
	if ( row >= rows ) return;
	const float* l = logits + static_cast<size_t>( row ) * experts;
	const int *B = sel + static_cast<size_t>( row ) * k, *A = tid + static_cast<size_t>( row ) * tk;
	float* d = dl + static_cast<size_t>( row ) * experts;
	float best = -INFINITY;
	int bi = -1;
	for ( int e = lane; e < experts; e += 32 ) {
		bool in = false;
		for ( int j = 0; j < k; ++j ) in = in || A[j] == e || B[j] == e;
		if ( !in && l[e] > best ) {
			best = l[e];
			bi = e;
		}
	}
	for ( int o = 16; o > 0; o >>= 1 ) {
		const float ob = __shfl_xor_sync( 0xffffffff, best, o );
		const int oi = __shfl_xor_sync( 0xffffffff, bi, o );
		if ( ob > best || ( ob == best && oi >= 0 && ( bi < 0 || oi < bi ) ) ) {
			best = ob;
			bi = oi;
		}
	}
	if ( lane != 0 ) return;
	auto in_set = []( const int* s, int n, int e ) {
		for ( int j = 0; j < n; ++j )
			if ( s[j] == e ) return true;
		return false;
	};
	auto softplus = []( float x ) { return x > 20.0f ? x : log1pf( expf( x ) ); };
	auto sigm = []( float x ) { return 1.0f / ( 1.0f + expf( -x ) ); };
	float ls = 0.0f;
	for ( int a = 0; a < k; ++a ) {
		const int w = B[a];
		if ( in_set( A, k, w ) ) continue;
		for ( int b = 0; b < k; ++b ) {
			const int m = A[b];
			if ( in_set( B, k, m ) ) continue;
			const float x = l[w] - l[m] + delta;
			ls += softplus( x );
			const float g = sigm( x );
			d[w] += g;
			d[m] -= g;
		}
	}
	int c = -1;
	for ( int a = 0; a < k; ++a )
		if ( in_set( B, k, A[a] ) && ( c < 0 || l[A[a]] < l[c] ) ) c = A[a];
	if ( c >= 0 && bi >= 0 ) {
		const float x = delta - ( l[c] - l[bi] );
		ls += softplus( x );
		const float g = sigm( x );
		d[c] -= g;
		d[bi] += g;
	}
	loss[row] = ls;
}

// Per row of ks = k + c slots: the selected ones' logit gradients through their softmax, dl = w ( dw - s ) with s = sum_j w_j dw_j (the routes'
// output against dy), and a candidate's wp ( dw - s ): the gradient of its logit were it in the softmax at a vanishing weight
__global__ void k_route_candidates_bwd( const float* sw, const float* dw, const float* wp, int rows, int k, int c, float* dl ) {
	Q4E_LOOP( r, (size_t) rows ) {
		const int ks = k + c;
		const float* wr = sw + r * ks;
		const float* dr = dw + r * ks;
		float s = 0.0f;
		for ( int j = 0; j < k; ++j ) s += wr[j] * dr[j];
		for ( int j = 0; j < k; ++j ) dl[r * ks + j] = wr[j] * ( dr[j] - s );
		for ( int j = 0; j < c; ++j ) dl[r * ks + k + j] = wp[r * c + j] * ( dr[k + j] - s );
	}
}

// ---- PLE gate, segment dots, column copies, routing pieces ----
__global__ void k_ple_gate( const float* g, float* sg, size_t n ) {
	Q4E_LOOP( i, n ) {
		const float x = g[i], sgn = x > 0.0f ? 1.0f : ( x < 0.0f ? -1.0f : 0.0f );
		sg[i] = sigmoidf_( sgn * sqrtf( fmaxf( fabsf( x ), 1e-6f ) ) );
	}
}

__global__ void k_ple_gate_bwd( const float* g, const float* dsg, float* dg, size_t n ) {
	Q4E_LOOP( i, n ) {
		const float x = g[i], sgn = x > 0.0f ? 1.0f : ( x < 0.0f ? -1.0f : 0.0f ), a = fabsf( x );
		const float ss = sgn * sqrtf( fmaxf( a, 1e-6f ) ), s = sigmoidf_( ss );
		dg[i] = a > 1e-6f ? dsg[i] * s * ( 1.0f - s ) * 0.5f / sqrtf( a ) : 0.0f;
	}
}

__global__ void k_seg_dot( const float* a, const float* b, int n, float scale, float* out ) {
	__shared__ float sh[32];
	const size_t base = static_cast<size_t>( blockIdx.x ) * n;
	float s = 0.0f;
	for ( int i = threadIdx.x; i < n; i += blockDim.x ) s += a[base + i] * b[base + i];
	s = block_sum256( s, sh );
	if ( threadIdx.x == 0 ) out[blockIdx.x] = s * scale;
}

__global__ void k_seg_dot_bwd( const float* a, const float* dout, int segments, int n, float scale, float* db ) {
	Q4E_LOOP( i, (size_t) segments * n ) db[i] += scale * dout[i / n] * a[i];
}

__global__ void k_copy_cols( const float* src, int src_ld, int src_col, float* dst, int dst_ld, int dst_col, int rows, int cols, bool add ) {
	Q4E_LOOP( i, (size_t) rows * cols ) {
		const size_t r = i / cols, c = i % cols;
		const float v = src[r * src_ld + src_col + c];
		float& d = dst[r * dst_ld + dst_col + c];
		d = add ? d + v : v;
	}
}

__global__ void k_half_to_float( const half* h, float* f, size_t n ) {
	Q4E_LOOP( i, n ) f[i] = __half2float( h[i] );
}

__global__ void k_round_half( float* x, size_t n ) {
	Q4E_LOOP( i, n ) x[i] = __half2float( __float2half_rn( x[i] ) );
}

// A warp per 128 values of a row: lane l holds values 4 l .. 4 l + 3; the Sylvester transform, a butterfly per index bit, scaled to unit norm
__global__ void k_had_rows( const float* in, const float* col_scale, int rows, int cols, float* out ) {
	const size_t w = ( static_cast<size_t>( blockIdx.x ) * blockDim.x + threadIdx.x ) / 32;
	const int lane = threadIdx.x % 32, blocks = cols / 128;
	if ( w >= static_cast<size_t>( rows ) * blocks ) return;
	const size_t r = w / blocks;
	const int c = static_cast<int>( w % blocks ) * 128 + lane * 4;
	float4 v = *reinterpret_cast<const float4*>( in + r * cols + c );
	if ( col_scale ) {
		const float4 s = *reinterpret_cast<const float4*>( col_scale + c );
		v.x *= s.x;
		v.y *= s.y;
		v.z *= s.z;
		v.w *= s.w;
	}
	float a0 = v.x + v.y, a1 = v.x - v.y, a2 = v.z + v.w, a3 = v.z - v.w;
	float x[4] = { a0 + a2, a1 + a3, a0 - a2, a1 - a3 };
	#pragma unroll
	for ( int b = 1; b < 32; b <<= 1 ) {
		#pragma unroll
		for ( int i = 0; i < 4; ++i ) {
			const float p = __shfl_xor_sync( 0xffffffff, x[i], b );
			x[i] = ( lane & b ) ? p - x[i] : x[i] + p;
		}
	}
	constexpr float s = 0.08838834764831845f;
	*reinterpret_cast<float4*>( out + r * cols + c ) = make_float4( x[0] * s, x[1] * s, x[2] * s, x[3] * s );
}

__global__ void k_gain_codes( const float* g, uint8_t* codes, size_t n ) {
	Q4E_LOOP( i, n ) codes[i] = static_cast<uint8_t>( fminf( fmaxf( rintf( g[i] * 1024.0f - 896.0f ), 0.0f ), 255.0f ) );
}

__global__ void k_gain_values( const uint8_t* codes, float* g, size_t n ) {
	Q4E_LOOP( i, n ) g[i] = ( 896.0f + codes[i] ) * ( 1.0f / 1024.0f );
}

__global__ void k_add_scaled( float* x, const float* y, float a, size_t n ) {
	Q4E_LOOP( i, n ) x[i] += a * y[i];
}

__global__ void k_router_input_bwd( const float* dlogit, const int* ids, const float* R, int rows, int k, int hidden, float* dX ) {
	Q4E_LOOP( i, (size_t) rows * hidden ) {
		const size_t r = i / hidden, d = i % hidden;
		float acc = 0.0f;
		for ( int j = 0; j < k; ++j ) acc += dlogit[r * k + j] * R[static_cast<size_t>( ids[r * k + j] ) * hidden + d];
		dX[i] += acc;
	}
}

// ---- grouped routed experts (rows in float4s: cols4 = cols / 4) ----
__device__ __forceinline__ float4 mul4( float4 a, float4 b ) { return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w ); }
__device__ __forceinline__ float4 add4( float4 a, float4 b ) { return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }

__global__ void k_route_gather( const float4* src, const int* prow, const int* pslot, const float* w, int P, int cols4, float4* dst ) {
	Q4E_LOOP( i, (size_t) P * cols4 ) {
		const size_t p = i / cols4;
		const int r = prow[p];
		if ( r < 0 ) {
			dst[i] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
			continue;
		}
		const float4 v = src[(size_t) r * cols4 + i % cols4];
		const float s = w ? w[pslot[p]] : 1.0f;
		dst[i] = make_float4( v.x * s, v.y * s, v.z * s, v.w * s );
	}
}

__global__ void k_route_scale( const float4* in, const int* gexp, int m, const float* const* tab, int P, int cols4, float4* out, bool add ) {
	Q4E_LOOP( i, (size_t) P * cols4 ) {
		const size_t p = i / cols4;
		const float4 s = reinterpret_cast<const float4*>( tab[gexp[p / m]] )[i % cols4];
		const float4 v = mul4( in[i], s );
		out[i] = add ? add4( out[i], v ) : v;
	}
}

// One block of 256 threads per padded row: in[p] * tab[gexp[p / m]] as FP16 at a power-of-two scale (the row's largest magnitude to
// [2^13, 2^14)), rinv[p] its inverse
__global__ void k_route_scale_half( const float* in, const int* gexp, int m, const float* const* tab, int cols, __half* out, float* rinv ) {
	__shared__ float red[8];
	const size_t p = blockIdx.x;
	const float* row = in + p * cols;
	const float* s = tab[gexp[p / m]];
	float mx = 0.0f;
	for ( int c = threadIdx.x; c < cols; c += blockDim.x ) mx = fmaxf( mx, fabsf( row[c] * s[c] ) );
	for ( int o = 16; o > 0; o >>= 1 ) mx = fmaxf( mx, __shfl_xor_sync( 0xffffffffu, mx, o ) );
	if ( ( threadIdx.x & 31 ) == 0 ) red[threadIdx.x >> 5] = mx;
	__syncthreads();
	mx = 0.0f;
	for ( int w = 0; w < ( blockDim.x >> 5 ); ++w ) mx = fmaxf( mx, red[w] );
	int e = 0;
	if ( mx > 0.0f && isfinite( mx ) ) {
		int ex = 0;
		frexpf( mx, &ex );
		e = min( 120, max( -120, 14 - ex ) );
	}
	const float sc = ldexpf( 1.0f, e );
	for ( int c = threadIdx.x; c < cols; c += blockDim.x ) out[p * cols + c] = __float2half_rn( row[c] * s[c] * sc );
	if ( threadIdx.x == 0 ) rinv[p] = ldexpf( 1.0f, -e );
}

__device__ __forceinline__ float silu_( float x ) { return x * sigmoidf_( x ); }

__global__ void k_route_swiglu( const float4* Zg, const float4* Zu, const int* gexp, int m, const float* const* svg, const float* const* svu, int P, int cols4,
	float4* G, float4* U, float4* A ) {
	Q4E_LOOP( i, (size_t) P * cols4 ) {
		const int e = gexp[i / cols4 / m];
		const size_t c = i % cols4;
		const float4 g = mul4( Zg[i], reinterpret_cast<const float4*>( svg[e] )[c] ), u = mul4( Zu[i], reinterpret_cast<const float4*>( svu[e] )[c] );
		G[i] = g;
		U[i] = u;
		A[i] = make_float4( silu_( g.x ) * u.x, silu_( g.y ) * u.y, silu_( g.z ) * u.z, silu_( g.w ) * u.w );
	}
}

__device__ __forceinline__ void swiglu_grad1( float g, float u, float da, float& dg, float& du ) {
	const float s = sigmoidf_( g );
	dg = da * u * s * ( 1.0f + g * ( 1.0f - s ) );
	du = da * g * s;
}

__global__ void k_route_swiglu_bwd( const float4* dAk, const float4* G, const float4* U, const int* gexp, int m, const float* const* sud, int P, int cols4,
	float4* dG, float4* dU ) {
	Q4E_LOOP( i, (size_t) P * cols4 ) {
		const int e = gexp[i / cols4 / m];
		const size_t c = i % cols4;
		const float4 da = mul4( dAk[i], reinterpret_cast<const float4*>( sud[e] )[c] ), g = G[i], u = U[i];
		float4 dg, du;
		swiglu_grad1( g.x, u.x, da.x, dg.x, du.x );
		swiglu_grad1( g.y, u.y, da.y, dg.y, du.y );
		swiglu_grad1( g.z, u.z, da.z, dg.z, du.z );
		swiglu_grad1( g.w, u.w, da.w, dg.w, du.w );
		dG[i] = dg;
		dU[i] = du;
	}
}

// Block (column chunk of 256, expert i of the group): out[gexp[i]][c] += sum over the expert's m rows of A B
__global__ void k_seg_colsum_prod( const float* A, const float* B, int m, int cols, const int* gexp, float* const* out ) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= cols ) return;
	const size_t r0 = (size_t) blockIdx.y * m;
	float s = 0.0f;
	for ( int r = 0; r < m; ++r ) s += A[( r0 + r ) * cols + c] * B[( r0 + r ) * cols + c];
	out[gexp[blockIdx.y]][c] += s;
}

// out[r] += sum_j w[r k + j] Y[inv[r k + j] - p0] over the slots whose padded row falls in [p0, p0 + P), in slot order
__global__ void k_route_combine( const float4* Y, const int* inv, int p0, int P, const float* w, int rows, int k, int cols4, float4* out ) {
	Q4E_LOOP( i, (size_t) rows * cols4 ) {
		const size_t r = i / cols4, c = i % cols4;
		float4 acc = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
		bool any = false;
		for ( int j = 0; j < k; ++j ) {
			const int q = inv[r * k + j] - p0;
			if ( q < 0 || q >= P ) continue;
			const float s = w ? w[r * k + j] : 1.0f;
			const float4 v = Y[(size_t) q * cols4 + c];
			acc = make_float4( acc.x + s * v.x, acc.y + s * v.y, acc.z + s * v.z, acc.w + s * v.w );
			any = true;
		}
		if ( any ) out[i] = add4( out[i], acc );
	}
}

// One block per padded row: out[pslot[p]] = dY[prow[p]] . Y[p] (padding rows skipped)
__global__ void k_route_dots_pad( const float* dY, const int* prow, const int* pslot, const float* Y, int cols, float* out ) {
	__shared__ float sh[32];
	const int p = blockIdx.x, r = prow[p];
	if ( r < 0 ) return;
	const float* a = dY + static_cast<size_t>( r ) * cols;
	const float* b = Y + static_cast<size_t>( p ) * cols;
	float s = 0.0f;
	for ( int c = threadIdx.x; c < cols; c += blockDim.x ) s += a[c] * b[c];
	s = block_sum256( s, sh );
	if ( threadIdx.x == 0 ) out[pslot[p]] = s;
}

// ---- KL head ----
// One block per row
__global__ void k_kl_lse( const float* logits, int nc, float* m, float* s ) {
	__shared__ float sh[32];
	const int r = blockIdx.x;
	const float* row = logits + static_cast<size_t>( r ) * nc;
	float mx = -INFINITY;
	for ( int i = threadIdx.x; i < nc; i += blockDim.x ) mx = fmaxf( mx, row[i] );
	for ( int o = 16; o > 0; o >>= 1 ) mx = fmaxf( mx, __shfl_xor_sync( 0xffffffffu, mx, o ) );
	__syncthreads();
	if ( ( threadIdx.x & 31 ) == 0 ) sh[threadIdx.x >> 5] = mx;
	__syncthreads();
	mx = -INFINITY;
	for ( int w = 0; w < ( blockDim.x >> 5 ); ++w ) mx = fmaxf( mx, sh[w] );
	const float m_new = fmaxf( m[r], mx );
	float sum = 0.0f;
	for ( int i = threadIdx.x; i < nc; i += blockDim.x ) sum += expf( row[i] - m_new );
	sum = block_sum256( sum, sh );
	if ( threadIdx.x == 0 ) {
		s[r] = s[r] * expf( m[r] - m_new ) + sum;
		m[r] = m_new;
	}
}

__global__ void k_kl_gather( float* logits, int rows, int nc, int c0, const int* ids, int tk, float* sl ) {
	Q4E_LOOP( i, (size_t) rows * tk ) {
		const int r = static_cast<int>( i / tk ), id = ids[i];
		if ( id >= c0 && id < c0 + nc ) {
			float& l = logits[static_cast<size_t>( r ) * nc + ( id - c0 )];
			sl[i] = l;
			l = -INFINITY;
		}
	}
}

// One thread per row. Bucket log-probabilities against the full LSE (the tail and the top entries combined stably); the gradients from
// log differences: q - p = p expm1( log q - log p ); then g = u - q sum( u ) over the buckets so that they sum to zero
__global__ void k_kl_finish( const float* m, const float* s, const float* sl, const float* p, const float* tail, int rows, int tk, float* lt, float* gq,
	float* ut, float* kl ) {
	Q4E_LOOP( r, (size_t) rows ) {
		const float ltr = s[r] > 0.0f ? m[r] + logf( s[r] ) : -INFINITY;
		float mx = ltr;
		for ( int j = 0; j < tk; ++j ) mx = fmaxf( mx, sl[r * tk + j] );
		float sum = ltr > -INFINITY ? expf( ltr - mx ) : 0.0f;
		for ( int j = 0; j < tk; ++j ) sum += expf( sl[r * tk + j] - mx );
		const float lse = mx + logf( sum );
		float k = 0.0f, usum = 0.0f;
		for ( int j = 0; j < tk; ++j ) {
			const float lq = sl[r * tk + j] - lse, pp = p[r * tk + j];
			float u;
			if ( pp > 0.0f ) {
				const float lp = logf( pp );
				k += pp * ( lp - lq );
				u = pp * expm1f( fminf( lq - lp, 80.0f ) );
			}
			else u = expf( lq );
			gq[r * tk + j] = u;
			usum += u;
		}
		const float lqt = ltr - lse, pt = tail[r];
		float u_t;
		if ( pt > 0.0f ) {
			const float lpt = logf( pt );
			k += pt * ( lpt - lqt );
			u_t = pt * expm1f( fminf( lqt - lpt, 80.0f ) );
		}
		else u_t = expf( lqt );
		usum += u_t;
		// zero sum: subtract q_bucket * sum( u )
		for ( int j = 0; j < tk; ++j ) gq[r * tk + j] -= expf( sl[r * tk + j] - lse ) * usum;
		ut[r] = u_t - expf( lqt ) * usum;
		lt[r] = ltr;
		kl[r] = k;
	}
}

__global__ void k_kl_lse_combine( const float* m_all, const float* s_all, int ranks, int rows, float* m, float* s ) {
	Q4E_LOOP( r, (size_t) rows ) {
		float mx = -INFINITY;
		for ( int g = 0; g < ranks; ++g ) mx = fmaxf( mx, m_all[(size_t) g * rows + r] );
		float sum = 0.0f;
		if ( mx > -INFINITY )
			for ( int g = 0; g < ranks; ++g ) {
				const float mg = m_all[(size_t) g * rows + r];
				if ( mg > -INFINITY ) sum += s_all[(size_t) g * rows + r] * expf( mg - mx );
			}
		m[r] = mx;
		s[r] = sum;
	}
}

__global__ void k_kl_grad_all( const float* logits, int rows, int nc, const float* lt, const float* ut, float* dl ) {
	Q4E_LOOP( i, (size_t) rows * nc ) {
		const size_t r = i / nc;
		dl[i] = expf( logits[i] - lt[r] ) * ut[r];
	}
}

__global__ void k_kl_grad_top( int rows, int nc, int c0, const int* ids, const float* gq, int tk, float* dl ) {
	Q4E_LOOP( i, (size_t) rows * tk ) {
		const int r = static_cast<int>( i / tk ), id = ids[i];
		if ( id >= c0 && id < c0 + nc ) dl[static_cast<size_t>( r ) * nc + ( id - c0 )] = gq[i];
	}
}

__global__ void k_mul_scaled( const float* x, const float* y, float s, float* out, size_t n ) {
	Q4E_LOOP( i, n ) out[i] = x[i] * y[i] * s;
}

__global__ void k_router_given( const float* logits, const int* ids, int rows, int experts, int k, float* w ) {
	Q4E_LOOP( r, (size_t) rows ) {
		float m = -INFINITY;
		for ( int j = 0; j < k; ++j ) m = fmaxf( m, logits[r * experts + ids[r * k + j]] );
		float s = 0.0f;
		for ( int j = 0; j < k; ++j ) s += expf( logits[r * experts + ids[r * k + j]] - m );
		for ( int j = 0; j < k; ++j ) w[r * k + j] = expf( logits[r * experts + ids[r * k + j]] - m ) / s;
	}
}

} // namespace

void rms_forward( const float* x, const float* w, int wrows, bool one_plus, bool mean, int segments, int n, float eps, float* inv, float* y,
	backend::stream_t stream ) {
	k_rms_fwd<<<segments, 256, 0, cu( stream )>>>( x, w, wrows, one_plus, mean, n, eps, inv, y );
}

void rms_backward( const float* x, const float* w, int wrows, bool one_plus, bool mean, int segments, int n, const float* inv, const float* dy, float* dx,
	backend::stream_t stream, float* dw ) {
	k_rms_bwd<<<segments, 256, 0, cu( stream )>>>( x, w, wrows, one_plus, mean, n, inv, dy, dx );
	if ( dw ) norm_dw( x, nullptr, inv, dy, segments, wrows, n, dw, cu( stream ) );   // d / dw of x inv ( 1 + w ) or x inv w: x inv
}

void hc_forward( Blas& b, const HcWeights& w, const float* x, int rows, const HcCache& c, float* mixed, float* post ) {
	const cudaStream_t st = cu( b.stream() );
	const int flat = w.hc * w.hidden;
	rms_forward( x, w.norm, w.hc, true, true, rows * w.hc, w.hidden, w.eps, c.inv, c.normed, b.stream() );
	b.gemm( false, true, rows, w.rank, flat, 1.0f / w.hc, c.normed, w.down, 0.0f, c.zt );
	k_silu<<<blocks_for( (size_t) rows * w.rank ), 256, 0, st>>>( c.zt, c.t, (size_t) rows * w.rank );
	if ( w.inject != nullptr ) {
		b.gemm( false, true, rows, w.hc, flat, 1.0f / w.hc, c.normed, w.inject, 0.0f, c.zi );
		k_post<<<blocks_for( (size_t) rows * w.hc ), 256, 0, st>>>( c.zi, post, (size_t) rows * w.hc );
	}
	b.gemm( false, true, rows, flat, w.rank, 1.0f, c.t, w.up, 0.0f, c.u );
	k_hc_mix<<<blocks_for( (size_t) rows * w.hidden ), 256, 0, st>>>( c.u, c.normed, rows, w.hidden, w.hc, mixed );
}

size_t hc_scratch_floats( const HcWeights& w, int rows ) {
	return 2 * static_cast<size_t>( rows ) * w.hc * w.hidden + static_cast<size_t>( rows ) * w.rank + static_cast<size_t>( rows ) * w.hc;
}

void hc_backward( Blas& b, const HcWeights& w, const float* x, int rows, const HcCache& c, const float* dmixed, const float* dpost, float* dx, float* scratch ) {
	const cudaStream_t st = cu( b.stream() );
	const int flat = w.hc * w.hidden;
	float* dnormed = scratch;
	float* du = dnormed + static_cast<size_t>( rows ) * flat;
	float* dt = du + static_cast<size_t>( rows ) * flat;
	float* dzi = dt + static_cast<size_t>( rows ) * w.rank;
	k_hc_mix_bwd<<<blocks_for( (size_t) rows * flat ), 256, 0, st>>>( c.u, c.normed, dmixed, rows, w.hidden, w.hc, dnormed, du );
	b.gemm( false, false, rows, w.rank, flat, 1.0f, du, w.up, 0.0f, dt );
	if ( w.gup ) b.gemm( true, false, flat, w.rank, rows, 1.0f, du, c.t, 1.0f, w.gup );   // u = t up^T
	k_silu_bwd<<<blocks_for( (size_t) rows * w.rank ), 256, 0, st>>>( c.zt, dt, (size_t) rows * w.rank );
	if ( w.gdown ) b.gemm( true, false, w.rank, flat, rows, 1.0f / w.hc, dt, c.normed, 1.0f, w.gdown );   // zt = normed down^T / hc
	b.gemm( false, false, rows, flat, w.rank, 1.0f / w.hc, dt, w.down, 1.0f, dnormed );
	if ( w.inject != nullptr && dpost != nullptr ) {
		k_post_bwd<<<blocks_for( (size_t) rows * w.hc ), 256, 0, st>>>( c.zi, dpost, dzi, (size_t) rows * w.hc );
		b.gemm( false, false, rows, flat, w.hc, 1.0f / w.hc, dzi, w.inject, 1.0f, dnormed );
		if ( w.ginject ) b.gemm( true, false, w.hc, flat, rows, 1.0f / w.hc, dzi, c.normed, 1.0f, w.ginject );
	}
	rms_backward( x, w.norm, w.hc, true, true, rows * w.hc, w.hidden, c.inv, dnormed, dx, b.stream(), w.gnorm );
}

void apply_forward( float* x, const float* post, const float* y, int rows, int hidden, int hc, backend::stream_t stream ) {
	k_apply<<<blocks_for( (size_t) rows * hc * hidden ), 256, 0, cu( stream )>>>( x, post, y, rows, hidden, hc );
}

void apply_backward( const float* dx, const float* post, const float* y, int rows, int hidden, int hc, float* dy, float* dpost, backend::stream_t stream ) {
	k_apply_bwd_dy<<<blocks_for( (size_t) rows * hidden ), 256, 0, cu( stream )>>>( dx, post, rows, hidden, hc, dy );
	k_apply_bwd_dpost<<<rows * hc, 256, 0, cu( stream )>>>( dx, y, hidden, hc, dpost );
}

void conv4_silu_forward( const float* x, const float* w, int rows, int channels, int dil, float* s, float* y, backend::stream_t stream, int seq_len ) {
	k_conv4_fwd<<<blocks_for( (size_t) rows * channels ), 256, 0, cu( stream )>>>( x, w, rows, channels, dil, seq_len > 0 ? seq_len : rows, s, y );
}

void conv4_silu_backward( const float* s, const float* w, int rows, int channels, int dil, const float* dy, float* dx, backend::stream_t stream,
	int seq_len ) {
	k_conv4_bwd<<<blocks_for( (size_t) rows * channels ), 256, 0, cu( stream )>>>( s, w, rows, channels, dil, seq_len > 0 ? seq_len : rows, dy, dx );
}

void conv4_silu_dw( const float* x, const float* s, const float* dy, int rows, int channels, int ld, int dil, float* dw, backend::stream_t stream,
	int seq_len ) {
	if ( channels > 0 && rows > 0 )
		k_conv4_dw<<<dim3( ( channels + 127 ) / 128, ( rows + kConvDwRows - 1 ) / kConvDwRows ), 128, 0, cu( stream )>>>( x, s, dy, rows, channels, ld, dil,
			seq_len > 0 ? seq_len : rows, dw );
}

void sigmoid_gate_forward( const float* a, const float* g, float* o, size_t n, backend::stream_t stream ) {
	k_sig_gate<<<blocks_for( n ), 256, 0, cu( stream )>>>( a, g, o, n );
}

void sigmoid_gate_backward( const float* a, const float* g, const float* dout, float* da, float* dg, size_t n, backend::stream_t stream ) {
	k_sig_gate_bwd<<<blocks_for( n ), 256, 0, cu( stream )>>>( a, g, dout, da, dg, n );
}

void gated_rms_forward( const float* x, const float* z, const float* gamma, int segments, int n, float eps, float* inv, float* y, backend::stream_t stream ) {
	k_grms_fwd<<<segments, 256, 0, cu( stream )>>>( x, z, gamma, n, eps, inv, y );
}

void gated_rms_backward( const float* x, const float* z, const float* gamma, int segments, int n, const float* inv, const float* dy, float* dx, float* dz,
	backend::stream_t stream, float* dgamma ) {
	k_grms_bwd<<<segments, 256, 0, cu( stream )>>>( x, z, gamma, n, inv, dy, dx, dz );
	if ( dgamma ) norm_dw( x, z, inv, dy, segments, 1, n, dgamma, cu( stream ) );
}

void topk_softmax_backward( const float* w, const float* dw, int rows, int k, float* dlogit, backend::stream_t stream ) {
	k_topk_softmax_bwd<<<blocks_for( rows ), 256, 0, cu( stream )>>>( w, dw, rows, k, dlogit );
}

void route_candidates( const float* logits, const int* ids, const float* w, int rows, int experts, int k, int c, int* sids, float* sw, float* wp,
	backend::stream_t stream ) {
	ENGINE_ASSERT( experts <= 1024, "route_candidates: up to 1024 experts" );
	if ( rows > 0 ) k_route_candidates<<<rows, ( experts + 31 ) / 32 * 32, 0, cu( stream )>>>( logits, ids, w, experts, k, c, sids, sw, wp );
}

void route_margin( const float* logits, const int* sel, const int* tid, int rows, int experts, int k, int tk, float delta, float* dl, float* loss,
	backend::stream_t stream ) {
	if ( rows > 0 ) k_route_margin<<<static_cast<unsigned>( ( static_cast<size_t>( rows ) * 32 + 255 ) / 256 ), 256, 0, cu( stream )>>>( logits, sel, tid, rows,
		experts, k, tk, delta, dl, loss );
}

void route_candidates_backward( const float* sw, const float* dw, const float* wp, int rows, int k, int c, float* dl, backend::stream_t stream ) {
	if ( rows > 0 ) k_route_candidates_bwd<<<blocks_for( rows ), 256, 0, cu( stream )>>>( sw, dw, wp, rows, k, c, dl );
}

void slots_to_dense( const float* v, const int* ids, int rows, int k, int experts, float* out, backend::stream_t stream ) {
	if ( rows > 0 ) k_slots_to_dense<<<blocks_for( (size_t) rows * k ), 256, 0, cu( stream )>>>( v, ids, rows, k, experts, out );
}

size_t gdn_ckpt_floats( int rows, int vh, int chunk, int seqs ) {
	return static_cast<size_t>( seqs ) * ( ( rows / seqs + chunk - 1 ) / chunk ) * vh * GD * GD;
}

size_t gdn_scratch_floats( int vh, int chunk, int seqs ) {
	return static_cast<size_t>( seqs ) * vh * GD * GD * ( ( chunk + kGdnSub - 1 ) / kGdnSub );
}

void gdn_forward( const float* q, const float* k, const float* v, const float* gate, const float* beta, int rows, int kh, int vh, float scale, int chunk,
	float* ckpt, float* o, backend::stream_t stream, int seqs ) {
	ENGINE_ASSERT( vh % kh == 0, "gdn_forward: value heads must be a multiple of the key heads" );
	ENGINE_ASSERT( seqs > 0 && rows % seqs == 0, "gdn_forward: the rows must hold whole sequences" );
	k_gdn_fwd<<<dim3( vh, GD / GW, seqs ), 256, 0, cu( stream )>>>( q, k, v, gate, beta, rows, kh, vh, scale, chunk, ckpt, o );
}

void gdn_backward( const float* q, const float* k, const float* v, const float* gate, const float* beta, int rows, int kh, int vh, float scale, int chunk,
	const float* ckpt, const float* dout, float* dq, float* dk, float* dv, float* dgate, float* dbeta, float* scratch, backend::stream_t stream, int seqs ) {
	ENGINE_ASSERT( seqs > 0 && rows % seqs == 0, "gdn_backward: the rows must hold whole sequences" );
	k_gdn_bwd<<<dim3( vh, GD / GW, seqs ), 256, 0, cu( stream )>>>( q, k, v, gate, beta, rows, kh, vh, scale, chunk, ckpt, dout, dq, dk, dv, dgate, dbeta,
		scratch );
}

void gdn_gates_forward( const float* a, const float* b, const float* A_log, const float* dt_bias, int rows, int vh, float* gate, float* beta,
	backend::stream_t stream ) {
	k_gdn_gates<<<blocks_for( (size_t) rows * vh ), 256, 0, cu( stream )>>>( a, b, A_log, dt_bias, rows, vh, gate, beta );
}

void gdn_gates_backward( const float* a, const float* b, const float* A_log, const float* dt_bias, int rows, int vh, const float* dgate, const float* dbeta,
	float* da, float* db, backend::stream_t stream ) {
	k_gdn_gates_bwd<<<blocks_for( (size_t) rows * vh ), 256, 0, cu( stream )>>>( a, b, A_log, dt_bias, rows, vh, dgate, dbeta, da, db );
}

void rope_forward( float* x, int rows, int heads, int head_dim, int rot_dim, float base, backend::stream_t stream, int seq_len ) {
	k_rope<<<blocks_for( (size_t) rows * heads * ( rot_dim / 2 ) ), 256, 0, cu( stream )>>>( x, rows, heads, head_dim, rot_dim, base, 1.0f,
		seq_len > 0 ? seq_len : rows );
}

void rope_backward( float* dx, int rows, int heads, int head_dim, int rot_dim, float base, backend::stream_t stream, int seq_len ) {
	k_rope<<<blocks_for( (size_t) rows * heads * ( rot_dim / 2 ) ), 256, 0, cu( stream )>>>( dx, rows, heads, head_dim, rot_dim, base, -1.0f,
		seq_len > 0 ? seq_len : rows );
}

void unpack_q_gate( const float* qg, int rows, int heads, int head_dim, float* q, float* gate, backend::stream_t stream ) {
	k_unpack_qg<<<blocks_for( (size_t) rows * heads * head_dim ), 256, 0, cu( stream )>>>( qg, rows, heads, head_dim, q, gate );
}

void pack_q_gate( const float* dq, const float* dgate, int rows, int heads, int head_dim, float* dqg, backend::stream_t stream ) {
	k_pack_qg<<<blocks_for( (size_t) rows * heads * head_dim ), 256, 0, cu( stream )>>>( dq, dgate, rows, heads, head_dim, dqg );
}

void attention_forward( Blas& b, const float* q, const float* k, const float* v, int rows, int hq, int hk, int d, float scale, float* s, float* o,
	const QsaSelection& sel, int qblock ) {
	const int group = hq / hk, qb = qblock > 0 ? std::min( qblock, rows ) : rows;
	for ( int r0 = 0; r0 < rows; r0 += qb ) {
		const int m = std::min( qb, rows - r0 ), n = r0 + m;   // the block's query rows and the keys they see
		for ( int h = 0; h < hq; ++h ) {
			const int kv = h / group;
			b.gemm( false, true, m, n, d, scale, q + static_cast<size_t>( r0 ) * hq * d + static_cast<size_t>( h ) * d, hq * d, k + static_cast<size_t>( kv ) * d,
				hk * d, 0.0f, s, n );
			k_attn_softmax<<<m, 256, 0, cu( b.stream() )>>>( s, n, r0, sel.bits, sel.words, sel.budget );
			b.gemm( false, false, m, d, n, 1.0f, s, n, v + static_cast<size_t>( kv ) * d, hk * d, 0.0f, o + static_cast<size_t>( r0 ) * hq * d + static_cast<size_t>( h ) * d,
				hq * d );
		}
	}
}

void attention_backward( Blas& b, const float* q, const float* k, const float* v, int rows, int hq, int hk, int d, float scale, const float* dout, float* dq,
	float* dk, float* dv, float* s, float* dp, const QsaSelection& sel, int qblock ) {
	const int group = hq / hk, qb = qblock > 0 ? std::min( qblock, rows ) : rows;
	for ( int r0 = 0; r0 < rows; r0 += qb ) {
		const int m = std::min( qb, rows - r0 ), n = r0 + m;
		const size_t qo = static_cast<size_t>( r0 ) * hq * d;
		for ( int h = 0; h < hq; ++h ) {
			const int kv = h / group;
			const float* qh = q + qo + static_cast<size_t>( h ) * d;
			const float* kh = k + static_cast<size_t>( kv ) * d;
			const float* vh = v + static_cast<size_t>( kv ) * d;
			const float* doh = dout + qo + static_cast<size_t>( h ) * d;
			b.gemm( false, true, m, n, d, scale, qh, hq * d, kh, hk * d, 0.0f, s, n );
			k_attn_softmax<<<m, 256, 0, cu( b.stream() )>>>( s, n, r0, sel.bits, sel.words, sel.budget );
			b.gemm( true, false, n, d, m, 1.0f, s, n, doh, hq * d, 1.0f, dv + static_cast<size_t>( kv ) * d, hk * d );
			b.gemm( false, true, m, n, d, 1.0f, doh, hq * d, vh, hk * d, 0.0f, dp, n );
			k_softmax_bwd<<<m, 256, 0, cu( b.stream() )>>>( s, dp, n );
			b.gemm( false, false, m, d, n, scale, dp, n, kh, hk * d, 1.0f, dq + qo + static_cast<size_t>( h ) * d, hq * d );
			b.gemm( true, false, n, d, m, scale, dp, n, qh, hq * d, 1.0f, dk + static_cast<size_t>( kv ) * d, hk * d );
		}
	}
}

void qsa_queries( const float* qk, int rows, int seq_len, const void* q_norm, int rot_dim, float theta, float eps, float* q, float* raw_k,
	backend::stream_t stream ) {
	k_qsa_q<<<dim3( rows, kQsaHeads + 1 ), 128, 0, cu( stream )>>>( qk, static_cast<const __half*>( q_norm ), seq_len, rot_dim, theta, eps, q, raw_k );
}

void qsa_pools( const float* raw_k, int seqs, int seq_len, const void* k_norm, int rot_dim, float theta, float eps, float* pools, backend::stream_t stream ) {
	const int np = seq_len / kQsaPool;
	if ( np > 0 ) k_qsa_pool<<<dim3( np, seqs ), 128, 0, cu( stream )>>>( raw_k, static_cast<const __half*>( k_norm ), seq_len, rot_dim, theta, eps, pools );
}

size_t qsa_scratch_floats( int seq_len, int qblock ) {
	return static_cast<size_t>( qblock ) * ( seq_len / kQsaPool );
}

void qsa_select( const float* q, const float* pools, int seq_len, int budget, int qblock, float* scratch, uint32_t* bits, backend::stream_t stream ) {
	const int np = seq_len / kQsaPool, words = ( np + 31 ) / 32;
	ENGINE_ASSERT( words <= kQsaMaxWords, "qsa_select: the sequence has more pools than the selection holds" );
	const float scale = 1.0f / std::sqrt( static_cast<float>( kQsaDim ) );
	for ( int r0 = 0; r0 < seq_len; r0 += qblock ) {
		const int m = std::min( qblock, seq_len - r0 );
		if ( np > 0 ) k_qsa_score<<<dim3( ( np + 127 ) / 128, m ), 128, 0, cu( stream )>>>( q, pools, r0, np, budget, scale, scratch );
		k_qsa_select<<<m, 1024, 0, cu( stream )>>>( scratch, np, r0, budget, words, bits );
	}
}

void router_topk( const float* logits, int rows, int experts, int k, int* ids, float* w, backend::stream_t stream ) {
	ENGINE_ASSERT( experts <= 1024 && k <= 16, "router_topk: up to 1024 experts and 16 routes" );
	k_router_topk<<<rows, ( experts + 31 ) / 32 * 32, 0, cu( stream )>>>( logits, experts, k, ids, w );
}

void ple_gate_forward( const float* g, float* sg, size_t n, backend::stream_t stream ) {
	k_ple_gate<<<blocks_for( n ), 256, 0, cu( stream )>>>( g, sg, n );
}

void ple_gate_backward( const float* g, const float* dsg, float* dg, size_t n, backend::stream_t stream ) {
	k_ple_gate_bwd<<<blocks_for( n ), 256, 0, cu( stream )>>>( g, dsg, dg, n );
}

void seg_dot( const float* a, const float* b, int segments, int n, float scale, float* out, backend::stream_t stream ) {
	k_seg_dot<<<segments, 256, 0, cu( stream )>>>( a, b, n, scale, out );
}

void seg_dot_backward_b( const float* a, const float* dout, int segments, int n, float scale, float* db, backend::stream_t stream ) {
	k_seg_dot_bwd<<<blocks_for( (size_t) segments * n ), 256, 0, cu( stream )>>>( a, dout, segments, n, scale, db );
}

void copy_cols( const float* src, int src_ld, int src_col, float* dst, int dst_ld, int dst_col, int rows, int cols, bool add, backend::stream_t stream ) {
	k_copy_cols<<<blocks_for( (size_t) rows * cols ), 256, 0, cu( stream )>>>( src, src_ld, src_col, dst, dst_ld, dst_col, rows, cols, add );
}

void half_to_float( const void* h, float* f, size_t n, backend::stream_t stream ) {
	k_half_to_float<<<blocks_for( n ), 256, 0, cu( stream )>>>( static_cast<const half*>( h ), f, n );
}

void had_rows( const float* in, const float* col_scale, int rows, int cols, float* out, backend::stream_t stream ) {
	const size_t threads = static_cast<size_t>( rows ) * ( cols / 128 ) * 32;
	if ( threads > 0 ) k_had_rows<<<static_cast<unsigned>( ( threads + 255 ) / 256 ), 256, 0, cu( stream )>>>( in, col_scale, rows, cols, out );
}

void gain_codes( const float* g, uint8_t* codes, size_t n, backend::stream_t stream ) {
	if ( n > 0 ) k_gain_codes<<<blocks_for( n ), 256, 0, cu( stream )>>>( g, codes, n );
}

void gain_values( const uint8_t* codes, float* g, size_t n, backend::stream_t stream ) {
	if ( n > 0 ) k_gain_values<<<blocks_for( n ), 256, 0, cu( stream )>>>( codes, g, n );
}

void add_scaled( float* x, const float* y, float a, size_t n, backend::stream_t stream ) {
	if ( n > 0 ) k_add_scaled<<<blocks_for( n ), 256, 0, cu( stream )>>>( x, y, a, n );
}

void round_to_half( float* x, size_t n, backend::stream_t stream ) {
	k_round_half<<<blocks_for( n ), 256, 0, cu( stream )>>>( x, n );
}

void router_input_backward( const float* dlogit, const int* ids, const float* R, int rows, int k, int hidden, float* dX, backend::stream_t stream ) {
	k_router_input_bwd<<<blocks_for( (size_t) rows * hidden ), 256, 0, cu( stream )>>>( dlogit, ids, R, rows, k, hidden, dX );
}

namespace {
const float4* f4( const float* p ) { return reinterpret_cast<const float4*>( p ); }
float4* f4( float* p ) { return reinterpret_cast<float4*>( p ); }
} // namespace

void route_gather( const float* src, const int* prow, const int* pslot, const float* w, int P, int cols, float* dst, backend::stream_t stream ) {
	ENGINE_ASSERT( cols % 4 == 0, "qwen4_exp training: route rows must be float4 multiples" );
	if ( P > 0 ) k_route_gather<<<blocks_for( (size_t) P * cols / 4 ), 256, 0, cu( stream )>>>( f4( src ), prow, pslot, w, P, cols / 4, f4( dst ) );
}

void route_scale( const float* in, const int* gexp, int m, const float* const* tab, int P, int cols, float* out, bool add, backend::stream_t stream ) {
	if ( P > 0 ) k_route_scale<<<blocks_for( (size_t) P * cols / 4 ), 256, 0, cu( stream )>>>( f4( in ), gexp, m, tab, P, cols / 4, f4( out ), add );
}

void route_scale_half( const float* in, const int* gexp, int m, const float* const* tab, int P, int cols, void* out, float* rinv, backend::stream_t stream ) {
	if ( P > 0 ) k_route_scale_half<<<P, 256, 0, cu( stream )>>>( in, gexp, m, tab, cols, static_cast<__half*>( out ), rinv );
}

void route_swiglu( const float* Zg, const float* Zu, const int* gexp, int m, const float* const* svg, const float* const* svu, int P, int cols, float* G, float* U,
	float* A, backend::stream_t stream ) {
	if ( P > 0 )
		k_route_swiglu<<<blocks_for( (size_t) P * cols / 4 ), 256, 0, cu( stream )>>>( f4( Zg ), f4( Zu ), gexp, m, svg, svu, P, cols / 4, f4( G ), f4( U ), f4( A ) );
}

void route_swiglu_backward( const float* dAk, const float* G, const float* U, const int* gexp, int m, const float* const* sud, int P, int cols, float* dG,
	float* dU, backend::stream_t stream ) {
	if ( P > 0 )
		k_route_swiglu_bwd<<<blocks_for( (size_t) P * cols / 4 ), 256, 0, cu( stream )>>>( f4( dAk ), f4( G ), f4( U ), gexp, m, sud, P, cols / 4, f4( dG ),
			f4( dU ) );
}

void seg_colsum_prod( const float* A, const float* B, int m, int cols, int experts, const int* gexp, float* const* out, backend::stream_t stream ) {
	if ( experts > 0 && m > 0 ) k_seg_colsum_prod<<<dim3( ( cols + 255 ) / 256, experts ), 256, 0, cu( stream )>>>( A, B, m, cols, gexp, out );
}

void route_combine( const float* Y, const int* inv, int p0, int P, const float* w, int rows, int k, int cols, float* out, backend::stream_t stream ) {
	k_route_combine<<<blocks_for( (size_t) rows * cols / 4 ), 256, 0, cu( stream )>>>( f4( Y ), inv, p0, P, w, rows, k, cols / 4, f4( out ) );
}

void route_dots_pad( const float* dY, const int* prow, const int* pslot, const float* Y, int P, int cols, float* out, backend::stream_t stream ) {
	if ( P > 0 ) k_route_dots_pad<<<P, 256, 0, cu( stream )>>>( dY, prow, pslot, Y, cols, out );
}

void kl_lse_update( const float* logits, int rows, int nc, float* m, float* s, backend::stream_t stream ) {
	k_kl_lse<<<rows, 256, 0, cu( stream )>>>( logits, nc, m, s );
}

void kl_gather( float* logits, int rows, int nc, int c0, const int* ids, int tk, float* sl, backend::stream_t stream ) {
	k_kl_gather<<<blocks_for( (size_t) rows * tk ), 256, 0, cu( stream )>>>( logits, rows, nc, c0, ids, tk, sl );
}

void kl_finish( const float* m, const float* s, const float* sl, const float* p, const float* tail, int rows, int tk, float* lt, float* gq, float* ut, float* kl,
	backend::stream_t stream ) {
	k_kl_finish<<<blocks_for( rows ), 256, 0, cu( stream )>>>( m, s, sl, p, tail, rows, tk, lt, gq, ut, kl );
}

void kl_lse_combine( const float* m_all, const float* s_all, int ranks, int rows, float* m, float* s, backend::stream_t stream ) {
	k_kl_lse_combine<<<blocks_for( rows ), 256, 0, cu( stream )>>>( m_all, s_all, ranks, rows, m, s );
}

void kl_grad( const float* logits, int rows, int nc, int c0, const float* lt, const float* ut, const int* ids, const float* gq, int tk, float* dl,
	backend::stream_t stream ) {
	k_kl_grad_all<<<blocks_for( (size_t) rows * nc ), 256, 0, cu( stream )>>>( logits, rows, nc, lt, ut, dl );
	k_kl_grad_top<<<blocks_for( (size_t) rows * tk ), 256, 0, cu( stream )>>>( rows, nc, c0, ids, gq, tk, dl );
}

void mul_scaled( const float* x, const float* y, float s, float* out, size_t n, backend::stream_t stream ) {
	k_mul_scaled<<<blocks_for( n ), 256, 0, cu( stream )>>>( x, y, s, out, n );
}

void router_given( const float* logits, const int* ids, int rows, int experts, int k, float* w, backend::stream_t stream ) {
	k_router_given<<<blocks_for( rows ), 256, 0, cu( stream )>>>( logits, ids, rows, experts, k, w );
}

} // namespace engine::train::qwen4_exp
