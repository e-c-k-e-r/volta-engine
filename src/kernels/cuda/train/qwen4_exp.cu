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

__global__ void k_conv4_fwd( const float* x, const float* w, int rows, int channels, int dil, float* s, float* y ) {
	Q4E_LOOP( i, (size_t) rows * channels ) {
		const int t = static_cast<int>( i / channels ), c = static_cast<int>( i % channels );
		float acc = 0.0f;
		for ( int j = 0; j < 4; ++j ) {
			const int src = t - ( 3 - j ) * dil;
			if ( src >= 0 ) acc += w[static_cast<size_t>( c ) * 4 + j] * x[static_cast<size_t>( src ) * channels + c];
		}
		s[i] = acc;
		y[i] = acc * sigmoidf_( acc );
	}
}

// dx[t, c] += sum_j w[c, j] ds[t + ( 3 - j ) dil, c], ds = dy silu'( s )
__global__ void k_conv4_bwd( const float* s, const float* w, int rows, int channels, int dil, const float* dy, float* dx ) {
	Q4E_LOOP( i, (size_t) rows * channels ) {
		const int t = static_cast<int>( i / channels ), c = static_cast<int>( i % channels );
		float acc = 0.0f;
		for ( int j = 0; j < 4; ++j ) {
			const int dst = t + ( 3 - j ) * dil;
			if ( dst < rows ) {
				const size_t o = static_cast<size_t>( dst ) * channels + c;
				const float sg = sigmoidf_( s[o] );
				acc += w[static_cast<size_t>( c ) * 4 + j] * dy[o] * sg * ( 1.0f + s[o] * ( 1.0f - sg ) );
			}
		}
		dx[i] += acc;
	}
}

// dw[c, j] += sum_t ds[t, c] x[t - ( 3 - j ) dil, c], ds = dy silu'( s ); x, s, dy rows of stride ld, channels c < n (one thread each)
__global__ void k_conv4_dw( const float* x, const float* s, const float* dy, int rows, int n, int ld, int dil, float* dw ) {
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if ( c >= n ) return;
	float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	for ( int t = 0; t < rows; ++t ) {
		const size_t o = static_cast<size_t>( t ) * ld + c;
		const float sg = sigmoidf_( s[o] ), ds = dy[o] * sg * ( 1.0f + s[o] * ( 1.0f - sg ) );
		for ( int j = 0; j < 4; ++j ) {
			const int src = t - ( 3 - j ) * dil;
			if ( src >= 0 ) acc[j] += ds * x[static_cast<size_t>( src ) * ld + c];
		}
	}
	for ( int j = 0; j < 4; ++j ) dw[static_cast<size_t>( c ) * 4 + j] += acc[j];
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

__device__ __forceinline__ float warp_sum_( float v ) {
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffffu, v, o );
	return v;
}

// Block (head, 16 columns), a warp per column holding its state row in 4 values per lane (element r * 32 + lane)
__global__ __launch_bounds__( GW * 32 )
void k_gdn_fwd( const float* q, const float* k, const float* v, const float* gate, const float* beta, int rows, int kh, int vh, float scale, int chunk,
	float* ckpt, float* o ) {
	const int h = blockIdx.x, c = blockIdx.y * GW + threadIdx.x / 32, lane = threadIdx.x & 31, khh = h / ( vh / kh );
	float S[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	for ( int t = 0; t < rows; ++t ) {
		if ( t % chunk == 0 ) {
			float* ck = ckpt + ( ( static_cast<size_t>( t / chunk ) * vh + h ) * GD + c ) * GD;
			for ( int r = 0; r < 4; ++r ) ck[r * 32 + lane] = S[r];
		}
		const float* kt = k + ( static_cast<size_t>( t ) * kh + khh ) * GD;
		const float* qt = q + ( static_cast<size_t>( t ) * kh + khh ) * GD;
		float kk[4], qq[4];
		for ( int r = 0; r < 4; ++r ) { kk[r] = kt[r * 32 + lane]; qq[r] = qt[r * 32 + lane]; }
		const float g = expf( gate[static_cast<size_t>( t ) * vh + h] ), bt = beta[static_cast<size_t>( t ) * vh + h];
		const float vc = v[( static_cast<size_t>( t ) * vh + h ) * GD + c];
		float u = 0.0f;
		for ( int r = 0; r < 4; ++r ) u += S[r] * kk[r];
		u = warp_sum_( u );
		const float d = bt * ( vc - g * u );
		float oc = 0.0f;
		for ( int r = 0; r < 4; ++r ) {
			S[r] = g * S[r] + d * kk[r];
			oc += S[r] * qq[r];
		}
		oc = warp_sum_( oc );
		if ( lane == 0 ) o[( static_cast<size_t>( t ) * vh + h ) * GD + c] = oc * scale;
	}
}

// One chunk of the backward: the chunk's states recomputed from its checkpoint into sb, then its rows in reverse with dS carried in carry
__global__ __launch_bounds__( GW * 32 )
void k_gdn_bwd( const float* q, const float* k, const float* v, const float* gate, const float* beta, int rows, int kh, int vh, float scale, int chunk,
	int n, const float* ckpt, const float* dout, float* dq, float* dk, float* dv, float* dgate, float* dbeta, float* sbuf, float* carry ) {
	__shared__ float s_dq[GD], s_dk[GD], s_g[2];
	const int h = blockIdx.x, c = blockIdx.y * GW + threadIdx.x / 32, lane = threadIdx.x & 31, khh = h / ( vh / kh );
	const int t0 = n * chunk, t1 = min( rows, t0 + chunk );
	float* sb = sbuf + ( static_cast<size_t>( h ) * GD + c ) * chunk * GD;
	float S[4];
	const float* ck = ckpt + ( ( static_cast<size_t>( n ) * vh + h ) * GD + c ) * GD;
	for ( int r = 0; r < 4; ++r ) S[r] = ck[r * 32 + lane];
	for ( int t = t0; t < t1; ++t ) {
		for ( int r = 0; r < 4; ++r ) sb[static_cast<size_t>( t - t0 ) * GD + r * 32 + lane] = S[r];
		const float* kt = k + ( static_cast<size_t>( t ) * kh + khh ) * GD;
		float kk[4];
		for ( int r = 0; r < 4; ++r ) kk[r] = kt[r * 32 + lane];
		const float g = expf( gate[static_cast<size_t>( t ) * vh + h] ), bt = beta[static_cast<size_t>( t ) * vh + h];
		const float vc = v[( static_cast<size_t>( t ) * vh + h ) * GD + c];
		float u = 0.0f;
		for ( int r = 0; r < 4; ++r ) u += S[r] * kk[r];
		u = warp_sum_( u );
		const float d = bt * ( vc - g * u );
		for ( int r = 0; r < 4; ++r ) S[r] = g * S[r] + d * kk[r];
	}
	float dS[4];
	float* cy = carry + ( static_cast<size_t>( h ) * GD + c ) * GD;
	for ( int r = 0; r < 4; ++r ) dS[r] = cy[r * 32 + lane];
	for ( int t = t1 - 1; t >= t0; --t ) {
		if ( threadIdx.x < GD ) { s_dq[threadIdx.x] = 0.0f; s_dk[threadIdx.x] = 0.0f; }
		if ( threadIdx.x < 2 ) s_g[threadIdx.x] = 0.0f;
		__syncthreads();
		const float* kt = k + ( static_cast<size_t>( t ) * kh + khh ) * GD;
		const float* qt = q + ( static_cast<size_t>( t ) * kh + khh ) * GD;
		float kk[4], qq[4], Sp[4], St[4];
		for ( int r = 0; r < 4; ++r ) {
			kk[r] = kt[r * 32 + lane];
			qq[r] = qt[r * 32 + lane];
			Sp[r] = sb[static_cast<size_t>( t - t0 ) * GD + r * 32 + lane];
		}
		const float g = expf( gate[static_cast<size_t>( t ) * vh + h] ), bt = beta[static_cast<size_t>( t ) * vh + h];
		const float vc = v[( static_cast<size_t>( t ) * vh + h ) * GD + c];
		float u = 0.0f;
		for ( int r = 0; r < 4; ++r ) u += Sp[r] * kk[r];
		u = warp_sum_( u );
		const float d = bt * ( vc - g * u );
		for ( int r = 0; r < 4; ++r ) St[r] = g * Sp[r] + d * kk[r];
		const float doc = dout[( static_cast<size_t>( t ) * vh + h ) * GD + c];
		for ( int r = 0; r < 4; ++r ) {
			dS[r] += scale * doc * qq[r];
			atomicAdd( &s_dq[r * 32 + lane], scale * St[r] * doc );
		}
		float dd = 0.0f, dg = 0.0f;
		for ( int r = 0; r < 4; ++r ) { dd += dS[r] * kk[r]; dg += dS[r] * Sp[r]; }
		dd = warp_sum_( dd );
		dg = warp_sum_( dg );
		const float du = -bt * g * dd;
		for ( int r = 0; r < 4; ++r ) {
			atomicAdd( &s_dk[r * 32 + lane], dS[r] * d + Sp[r] * du );
			dS[r] = g * dS[r] + du * kk[r];
		}
		if ( lane == 0 ) {
			dv[( static_cast<size_t>( t ) * vh + h ) * GD + c] += bt * dd;
			atomicAdd( &s_g[0], dd * ( vc - g * u ) );
			atomicAdd( &s_g[1], dg - bt * dd * u );
		}
		__syncthreads();
		if ( threadIdx.x < GD ) {
			atomicAdd( &dq[( static_cast<size_t>( t ) * kh + khh ) * GD + threadIdx.x], s_dq[threadIdx.x] );
			atomicAdd( &dk[( static_cast<size_t>( t ) * kh + khh ) * GD + threadIdx.x], s_dk[threadIdx.x] );
		}
		if ( threadIdx.x == 0 ) {
			atomicAdd( &dbeta[static_cast<size_t>( t ) * vh + h], s_g[0] );
			atomicAdd( &dgate[static_cast<size_t>( t ) * vh + h], s_g[1] * g );
		}
		__syncthreads();
	}
	for ( int r = 0; r < 4; ++r ) cy[r * 32 + lane] = dS[r];
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
__global__ void k_rope( float* x, int rows, int heads, int head_dim, int rot_dim, float base, float sign ) {
	const int pairs = rot_dim / 2;
	Q4E_LOOP( i, (size_t) rows * heads * pairs ) {
		const int p = static_cast<int>( i % pairs );
		const size_t rh = i / pairs;
		const int r = static_cast<int>( rh / heads );
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
__global__ void k_causal_softmax( float* s, int rows ) {
	__shared__ float sh[32];
	const int i = blockIdx.x;
	float* row = s + static_cast<size_t>( i ) * rows;
	float m = -INFINITY;
	for ( int j = threadIdx.x; j <= i; j += blockDim.x ) m = fmaxf( m, row[j] );
	for ( int o = 16; o > 0; o >>= 1 ) m = fmaxf( m, __shfl_xor_sync( 0xffffffffu, m, o ) );
	__syncthreads();
	if ( ( threadIdx.x & 31 ) == 0 ) sh[threadIdx.x >> 5] = m;
	__syncthreads();
	m = -INFINITY;
	for ( int w = 0; w < ( blockDim.x >> 5 ); ++w ) m = fmaxf( m, sh[w] );
	float sum = 0.0f;
	for ( int j = threadIdx.x; j <= i; j += blockDim.x ) sum += expf( row[j] - m );
	sum = block_sum256( sum, sh );
	const float inv = 1.0f / sum;
	for ( int j = threadIdx.x; j < rows; j += blockDim.x ) row[j] = j <= i ? expf( row[j] - m ) * inv : 0.0f;
}

// dp <- p ( dp - sum_j p dp ), per row
__global__ void k_softmax_bwd( const float* p, float* dp, int rows ) {
	__shared__ float sh[32];
	const int i = blockIdx.x;
	const float* pr = p + static_cast<size_t>( i ) * rows;
	float* dr = dp + static_cast<size_t>( i ) * rows;
	float s = 0.0f;
	for ( int j = threadIdx.x; j <= i; j += blockDim.x ) s += pr[j] * dr[j];
	s = block_sum256( s, sh );
	for ( int j = threadIdx.x; j < rows; j += blockDim.x ) dr[j] = j <= i ? pr[j] * ( dr[j] - s ) : 0.0f;
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
	k_silu_bwd<<<blocks_for( (size_t) rows * w.rank ), 256, 0, st>>>( c.zt, dt, (size_t) rows * w.rank );
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

void conv4_silu_forward( const float* x, const float* w, int rows, int channels, int dil, float* s, float* y, backend::stream_t stream ) {
	k_conv4_fwd<<<blocks_for( (size_t) rows * channels ), 256, 0, cu( stream )>>>( x, w, rows, channels, dil, s, y );
}

void conv4_silu_backward( const float* s, const float* w, int rows, int channels, int dil, const float* dy, float* dx, backend::stream_t stream ) {
	k_conv4_bwd<<<blocks_for( (size_t) rows * channels ), 256, 0, cu( stream )>>>( s, w, rows, channels, dil, dy, dx );
}

void conv4_silu_dw( const float* x, const float* s, const float* dy, int rows, int channels, int ld, int dil, float* dw, backend::stream_t stream ) {
	if ( channels > 0 ) k_conv4_dw<<<( channels + 127 ) / 128, 128, 0, cu( stream )>>>( x, s, dy, rows, channels, ld, dil, dw );
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

size_t gdn_ckpt_floats( int rows, int vh, int chunk ) {
	return static_cast<size_t>( ( rows + chunk - 1 ) / chunk ) * vh * GD * GD;
}

size_t gdn_scratch_floats( int vh, int chunk ) {
	return static_cast<size_t>( vh ) * GD * GD * ( chunk + 1 );
}

void gdn_forward( const float* q, const float* k, const float* v, const float* gate, const float* beta, int rows, int kh, int vh, float scale, int chunk,
	float* ckpt, float* o, backend::stream_t stream ) {
	ENGINE_ASSERT( vh % kh == 0, "gdn_forward: value heads must be a multiple of the key heads" );
	k_gdn_fwd<<<dim3( vh, GD / GW ), GW * 32, 0, cu( stream )>>>( q, k, v, gate, beta, rows, kh, vh, scale, chunk, ckpt, o );
}

void gdn_backward( const float* q, const float* k, const float* v, const float* gate, const float* beta, int rows, int kh, int vh, float scale, int chunk,
	const float* ckpt, const float* dout, float* dq, float* dk, float* dv, float* dgate, float* dbeta, float* scratch, backend::stream_t stream ) {
	float* sbuf = scratch;
	float* carry = scratch + static_cast<size_t>( vh ) * GD * GD * chunk;
	CUDA_CHECK( cudaMemsetAsync( carry, 0, static_cast<size_t>( vh ) * GD * GD * sizeof( float ), cu( stream ) ) );
	for ( int n = ( rows + chunk - 1 ) / chunk - 1; n >= 0; --n ) {
		k_gdn_bwd<<<dim3( vh, GD / GW ), GW * 32, 0, cu( stream )>>>( q, k, v, gate, beta, rows, kh, vh, scale, chunk, n, ckpt, dout, dq, dk, dv, dgate, dbeta,
			sbuf, carry );
	}
}

void gdn_gates_forward( const float* a, const float* b, const float* A_log, const float* dt_bias, int rows, int vh, float* gate, float* beta,
	backend::stream_t stream ) {
	k_gdn_gates<<<blocks_for( (size_t) rows * vh ), 256, 0, cu( stream )>>>( a, b, A_log, dt_bias, rows, vh, gate, beta );
}

void gdn_gates_backward( const float* a, const float* b, const float* A_log, const float* dt_bias, int rows, int vh, const float* dgate, const float* dbeta,
	float* da, float* db, backend::stream_t stream ) {
	k_gdn_gates_bwd<<<blocks_for( (size_t) rows * vh ), 256, 0, cu( stream )>>>( a, b, A_log, dt_bias, rows, vh, dgate, dbeta, da, db );
}

void rope_forward( float* x, int rows, int heads, int head_dim, int rot_dim, float base, backend::stream_t stream ) {
	k_rope<<<blocks_for( (size_t) rows * heads * ( rot_dim / 2 ) ), 256, 0, cu( stream )>>>( x, rows, heads, head_dim, rot_dim, base, 1.0f );
}

void rope_backward( float* dx, int rows, int heads, int head_dim, int rot_dim, float base, backend::stream_t stream ) {
	k_rope<<<blocks_for( (size_t) rows * heads * ( rot_dim / 2 ) ), 256, 0, cu( stream )>>>( dx, rows, heads, head_dim, rot_dim, base, -1.0f );
}

void unpack_q_gate( const float* qg, int rows, int heads, int head_dim, float* q, float* gate, backend::stream_t stream ) {
	k_unpack_qg<<<blocks_for( (size_t) rows * heads * head_dim ), 256, 0, cu( stream )>>>( qg, rows, heads, head_dim, q, gate );
}

void pack_q_gate( const float* dq, const float* dgate, int rows, int heads, int head_dim, float* dqg, backend::stream_t stream ) {
	k_pack_qg<<<blocks_for( (size_t) rows * heads * head_dim ), 256, 0, cu( stream )>>>( dq, dgate, rows, heads, head_dim, dqg );
}

void attention_forward( Blas& b, const float* q, const float* k, const float* v, int rows, int hq, int hk, int d, float scale, float* s, float* o ) {
	const int group = hq / hk;
	for ( int h = 0; h < hq; ++h ) {
		const int kv = h / group;
		b.gemm( false, true, rows, rows, d, scale, q + static_cast<size_t>( h ) * d, hq * d, k + static_cast<size_t>( kv ) * d, hk * d, 0.0f, s, rows );
		k_causal_softmax<<<rows, 256, 0, cu( b.stream() )>>>( s, rows );
		b.gemm( false, false, rows, d, rows, 1.0f, s, rows, v + static_cast<size_t>( kv ) * d, hk * d, 0.0f, o + static_cast<size_t>( h ) * d, hq * d );
	}
}

void attention_backward( Blas& b, const float* q, const float* k, const float* v, int rows, int hq, int hk, int d, float scale, const float* dout, float* dq,
	float* dk, float* dv, float* s, float* dp ) {
	const int group = hq / hk;
	for ( int h = 0; h < hq; ++h ) {
		const int kv = h / group;
		const float* qh = q + static_cast<size_t>( h ) * d;
		const float* kh = k + static_cast<size_t>( kv ) * d;
		const float* vh = v + static_cast<size_t>( kv ) * d;
		const float* doh = dout + static_cast<size_t>( h ) * d;
		b.gemm( false, true, rows, rows, d, scale, qh, hq * d, kh, hk * d, 0.0f, s, rows );
		k_causal_softmax<<<rows, 256, 0, cu( b.stream() )>>>( s, rows );
		b.gemm( true, false, rows, d, rows, 1.0f, s, rows, doh, hq * d, 1.0f, dv + static_cast<size_t>( kv ) * d, hk * d );
		b.gemm( false, true, rows, rows, d, 1.0f, doh, hq * d, vh, hk * d, 0.0f, dp, rows );
		k_softmax_bwd<<<rows, 256, 0, cu( b.stream() )>>>( s, dp, rows );
		b.gemm( false, false, rows, d, rows, scale, dp, rows, kh, hk * d, 1.0f, dq + static_cast<size_t>( h ) * d, hq * d );
		b.gemm( true, false, rows, d, rows, scale, dp, rows, qh, hq * d, 1.0f, dk + static_cast<size_t>( kv ) * d, hk * d );
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
