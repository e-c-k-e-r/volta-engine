// Training forms of GLM-5.3-Flash operations: the mHC site (engine/train/glm5_mhc.h). The pre step's forward is the mix product (Blas), a block
// per row for rmr and the pre / post gates, a thread per row for the Sinkhorn and a block per row for the collapse and its norm; its backward a
// block per row for the gates' gradients through the norm, a thread per row for the gates' and the Sinkhorn's backward (the iterates kept in the
// scratch, step-major so that a warp's rows are coalesced), the products back through fn (Blas), and the streams' gradient by columns.
#include "engine/train/glm5_mhc.h"
#include "engine/config.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace engine::train::glm5_next {

namespace {

#define MHC_LOOP( i, n ) for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ( n ); i += (size_t) gridDim.x * blockDim.x )

constexpr int kThreads = 256, kWarps = kThreads / 32;
constexpr int kRowThreads = 32;   // the thread-per-row kernels: one warp a block, so that the rows spread over the SMs

unsigned blocks_for( size_t n ) { return static_cast<unsigned>( ( n + 255 ) / 256 < 65535u * 8u ? ( n + 255 ) / 256 : 65535u * 8u ); }

cudaStream_t cu( backend::stream_t s ) { return static_cast<cudaStream_t>( s ); }

__device__ __forceinline__ float sigmoidf_( float x ) { return 1.0f / ( 1.0f + expf( -x ) ); }

// N sums over a block of kThreads threads, every thread getting them
template <int N>
__device__ __forceinline__ void block_sums( float ( &v )[N] ) {
	__shared__ float sh[N][kWarps];
#pragma unroll
	for ( int i = 0; i < N; ++i ) for ( int o = 16; o > 0; o >>= 1 ) v[i] += __shfl_xor_sync( 0xffffffffu, v[i], o );
	const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
	__syncthreads();   // a previous call's reads done
	if ( l == 0 ) {
#pragma unroll
		for ( int i = 0; i < N; ++i ) sh[i][w] = v[i];
	}
	__syncthreads();
#pragma unroll
	for ( int i = 0; i < N; ++i ) {
		float t = 0.0f;
		for ( int j = 0; j < kWarps; ++j ) t += sh[i][j];
		v[i] = t;
	}
}

// ---- Sinkhorn ----

// The rows' softmax of e [4][4] (g-major)
__device__ __forceinline__ void sk_softmax( const float ( &e )[kHcComb], float ( &p )[kHcComb] ) {
#pragma unroll
	for ( int g = 0; g < kHc; ++g ) {
		const float m = fmaxf( fmaxf( e[g * 4], e[g * 4 + 1] ), fmaxf( e[g * 4 + 2], e[g * 4 + 3] ) );
		float x[kHc], s = 0.0f;
#pragma unroll
		for ( int k = 0; k < kHc; ++k ) {
			x[k] = expf( e[g * 4 + k] - m );
			s += x[k];
		}
#pragma unroll
		for ( int k = 0; k < kHc; ++k ) p[g * 4 + k] = x[k] / s;
	}
}

// One normalization of v: each row (COL false) or column over ( its sum + eps )
template <bool COL>
__device__ __forceinline__ void sk_norm( float ( &v )[kHcComb], float eps ) {
#pragma unroll
	for ( int a = 0; a < kHc; ++a ) {
		float s = 0.0f;
#pragma unroll
		for ( int b = 0; b < kHc; ++b ) s += v[COL ? b * 4 + a : a * 4 + b];
		const float d = s + eps;
#pragma unroll
		for ( int b = 0; b < kHc; ++b ) v[COL ? b * 4 + a : a * 4 + b] /= d;
	}
}

// Its backward at the input x: y = x / d, d = sum( x ) + eps per group, so dx = ( dy - sum( dy y ) ) / d; dv holds dy in, dx out
template <bool COL>
__device__ __forceinline__ void sk_norm_bwd( const float ( &x )[kHcComb], float ( &dv )[kHcComb], float eps ) {
#pragma unroll
	for ( int a = 0; a < kHc; ++a ) {
		float s = 0.0f;
#pragma unroll
		for ( int b = 0; b < kHc; ++b ) s += x[COL ? b * 4 + a : a * 4 + b];
		const float d = s + eps;
		float t = 0.0f;
#pragma unroll
		for ( int b = 0; b < kHc; ++b ) {
			const int i = COL ? b * 4 + a : a * 4 + b;
			t += dv[i] * ( x[i] / d );
		}
#pragma unroll
		for ( int b = 0; b < kHc; ++b ) {
			const int i = COL ? b * 4 + a : a * 4 + b;
			dv[i] = ( dv[i] - t ) / d;
		}
	}
}

// The comb v of a row from its logits e. With st, the input of every normalization n (2 iters - 1 of them: n even a column one, odd a row
// one) is kept at st[( n * kHcComb + i ) * stride]
__device__ __forceinline__ void sinkhorn( const float ( &e )[kHcComb], int iters, float eps, float ( &v )[kHcComb], float* st, size_t stride ) {
	sk_softmax( e, v );
#pragma unroll
	for ( int i = 0; i < kHcComb; ++i ) v[i] += eps;
	size_t n = 0;
	auto keep = [&]() {
		if ( st != nullptr ) {
#pragma unroll
			for ( int i = 0; i < kHcComb; ++i ) st[( n * kHcComb + i ) * stride] = v[i];
		}
		++n;
	};
	keep();
	sk_norm<true>( v, eps );
	for ( int it = 1; it < iters; ++it ) {
		keep();
		sk_norm<false>( v, eps );
		keep();
		sk_norm<true>( v, eps );
	}
}

// The logits' gradient de from the comb's dcomb: the forward run again keeping its iterates in st, then every normalization backward from
// the last, then the softmax's (the + eps passes the gradient through)
__device__ __forceinline__ void sinkhorn_bwd( const float ( &e )[kHcComb], const float* dcomb, int iters, float eps, float* st, size_t stride,
	float ( &de )[kHcComb] ) {
	float v[kHcComb];
	sinkhorn( e, iters, eps, v, st, stride );
	float dv[kHcComb];
#pragma unroll
	for ( int i = 0; i < kHcComb; ++i ) dv[i] = dcomb[i];
	for ( int n = 2 * iters - 2; n >= 0; --n ) {
		float x[kHcComb];
#pragma unroll
		for ( int i = 0; i < kHcComb; ++i ) x[i] = st[( static_cast<size_t>( n ) * kHcComb + i ) * stride];
		if ( n % 2 == 0 ) sk_norm_bwd<true>( x, dv, eps );
		else sk_norm_bwd<false>( x, dv, eps );
	}
	float p[kHcComb];
	sk_softmax( e, p );
#pragma unroll
	for ( int g = 0; g < kHc; ++g ) {
		float t = 0.0f;
#pragma unroll
		for ( int k = 0; k < kHc; ++k ) t += p[g * 4 + k] * dv[g * 4 + k];
#pragma unroll
		for ( int k = 0; k < kHc; ++k ) de[g * 4 + k] = p[g * 4 + k] * ( dv[g * 4 + k] - t );
	}
}

// The comb logits of row r: E_i = l_{8+i} scale[2] + base[8+i]
__device__ __forceinline__ void comb_logits( const float* mix, float rr, const float* base, const float* scale, float ( &e )[kHcComb] ) {
	const float s2 = scale[2];
#pragma unroll
	for ( int i = 0; i < kHcComb; ++i ) e[i] = mix[2 * kHc + i] * rr * s2 + base[2 * kHc + i];
}

// ---- pre step forward ----

// Block per row: rmr, pre and post from the mix logits
__global__ void k_pre_gates( const float* X, const float* mix, const float* base, const float* scale, int hidden, float rms_eps, float hc_eps, float* rmr,
	float* pre, float* post ) {
	const int r = blockIdx.x, n = kHc * hidden;
	const float* x = X + static_cast<size_t>( r ) * n;
	float s[1] = { 0.0f };
	for ( int i = threadIdx.x; i < n; i += kThreads ) s[0] += x[i] * x[i];
	block_sums( s );
	const float rr = 1.0f / sqrtf( s[0] / n + rms_eps );
	if ( threadIdx.x == 0 ) rmr[r] = rr;
	const int t = threadIdx.x;
	if ( t < kHc ) pre[r * kHc + t] = sigmoidf_( mix[r * kHcMix + t] * rr * scale[0] + base[t] ) + hc_eps;
	else if ( t < 2 * kHc ) post[r * kHc + t - kHc] = 2.0f * sigmoidf_( mix[r * kHcMix + t] * rr * scale[1] + base[t] );
}

// Thread per row: comb
__global__ void k_pre_comb( const float* mix, const float* rmr, const float* base, const float* scale, int rows, int iters, float hc_eps, float* comb ) {
	const int r = blockIdx.x * blockDim.x + threadIdx.x;
	if ( r >= rows ) return;
	float e[kHcComb], v[kHcComb];
	comb_logits( mix + static_cast<size_t>( r ) * kHcMix, rmr[r], base, scale, e );
	sinkhorn( e, iters, hc_eps, v, nullptr, 0 );
#pragma unroll
	for ( int i = 0; i < kHcComb; ++i ) comb[static_cast<size_t>( r ) * kHcComb + i] = v[i];
}

// Block per row: u = sum_h pre_h X_h and a = u inv norm_w (u recomputed in the second pass)
__global__ void k_pre_collapse( const float* X, const float* pre, const float* w, int hidden, float norm_eps, float* inv, float* a ) {
	const int r = blockIdx.x;
	const float* x = X + static_cast<size_t>( r ) * kHc * hidden;
	float p[kHc];
#pragma unroll
	for ( int h = 0; h < kHc; ++h ) p[h] = pre[r * kHc + h];
	float s[1] = { 0.0f };
	for ( int d = threadIdx.x; d < hidden; d += kThreads ) {
		float u = 0.0f;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) u += p[h] * x[h * hidden + d];
		s[0] += u * u;
	}
	block_sums( s );
	const float iv = 1.0f / sqrtf( s[0] / hidden + norm_eps );
	if ( threadIdx.x == 0 ) inv[r] = iv;
	for ( int d = threadIdx.x; d < hidden; d += kThreads ) {
		float u = 0.0f;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) u += p[h] * x[h * hidden + d];
		a[static_cast<size_t>( r ) * hidden + d] = u * iv * w[d];
	}
}

// ---- pre step backward ----

// Block per row: with g = da norm_w and s = g . u, du = inv g - u inv^3 s / hidden, so dpre_h = du . X_h = inv ( g . X_h ) - inv^3 s / hidden
// ( u . X_h ); s kept for the streams' pass
__global__ void k_pre_bwd_row( const float* X, const float* da, const float* w, const float* pre, const float* inv, int hidden, float* srow, float* dpre ) {
	const int r = blockIdx.x;
	const float* x = X + static_cast<size_t>( r ) * kHc * hidden;
	const float* g = da + static_cast<size_t>( r ) * hidden;
	float p[kHc];
#pragma unroll
	for ( int h = 0; h < kHc; ++h ) p[h] = pre[r * kHc + h];
	float v[1 + 2 * kHc];   // s, g . X_h, u . X_h
#pragma unroll
	for ( int i = 0; i < 1 + 2 * kHc; ++i ) v[i] = 0.0f;
	for ( int d = threadIdx.x; d < hidden; d += kThreads ) {
		float xh[kHc], u = 0.0f;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) {
			xh[h] = x[h * hidden + d];
			u += p[h] * xh[h];
		}
		const float gd = g[d] * w[d];
		v[0] += gd * u;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) {
			v[1 + h] += gd * xh[h];
			v[1 + kHc + h] += u * xh[h];
		}
	}
	block_sums( v );
	if ( threadIdx.x == 0 ) {
		const float iv = inv[r], c3 = iv * iv * iv * v[0] / hidden;
		srow[r] = v[0];
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) dpre[r * kHc + h] = iv * v[1 + h] - c3 * v[1 + kHc + h];
	}
}

// Thread per row: the logits' gradients dz (pre and post through their sigmoids, comb through the Sinkhorn), dl = dz scale, dmix = dl rmr for
// the products, and the row coefficient crow = -( dl . mix ) rmr^3 / ( kHc hidden ) of rmr's gradient into X (d rmr / dx = -rmr^3 x / n)
__global__ void k_pre_bwd_gates( const float* mix, const float* rmr, const float* base, const float* scale, const float* dpre, const float* dpost,
	const float* dcomb, int rows, int hidden, int iters, float hc_eps, float* dz, float* dmix, float* crow, float* st ) {
	const int r = blockIdx.x * blockDim.x + threadIdx.x;
	if ( r >= rows ) return;
	const float* m = mix + static_cast<size_t>( r ) * kHcMix;
	const float rr = rmr[r];
	float z[kHcMix];
#pragma unroll
	for ( int h = 0; h < kHc; ++h ) {
		const float sp = sigmoidf_( m[h] * rr * scale[0] + base[h] );
		z[h] = dpre[r * kHc + h] * sp * ( 1.0f - sp );
		const float sq = sigmoidf_( m[kHc + h] * rr * scale[1] + base[kHc + h] );
		z[kHc + h] = dpost != nullptr ? dpost[r * kHc + h] * 2.0f * sq * ( 1.0f - sq ) : 0.0f;
	}
	if ( dcomb != nullptr ) {
		float e[kHcComb], de[kHcComb];
		comb_logits( m, rr, base, scale, e );
		sinkhorn_bwd( e, dcomb + static_cast<size_t>( r ) * kHcComb, iters, hc_eps, st + r, rows, de );
#pragma unroll
		for ( int i = 0; i < kHcComb; ++i ) z[2 * kHc + i] = de[i];
	}
	else {
#pragma unroll
		for ( int i = 0; i < kHcComb; ++i ) z[2 * kHc + i] = 0.0f;
	}
	float drmr = 0.0f;
#pragma unroll
	for ( int j = 0; j < kHcMix; ++j ) {
		const float dl = z[j] * scale[j < kHc ? 0 : j < 2 * kHc ? 1 : 2];
		dz[static_cast<size_t>( r ) * kHcMix + j] = z[j];
		dmix[static_cast<size_t>( r ) * kHcMix + j] = dl * rr;
		drmr += dl * m[j];
	}
	crow[r] = -drmr * rr * rr * rr / ( kHc * hidden );
}

// Thread per column d over a chunk of rows: dX_h[d] += pre_h du_d + crow x_h[d] (du as k_pre_bwd_row, u recomputed), and the norm weight's
// gradient sum_r da u inv (the chunks' sums added atomically)
__global__ void k_pre_bwd_dx( const float* X, const float* da, const float* w, const float* pre, const float* inv, const float* srow, const float* crow,
	int rows, int hidden, int per_chunk, float* dX, float* gnorm ) {
	const int d = blockIdx.x * blockDim.x + threadIdx.x;
	if ( d >= hidden ) return;
	const int r0 = blockIdx.y * per_chunk, r1 = min( rows, r0 + per_chunk );
	const float wd = w[d];
	float acc = 0.0f;
	for ( int r = r0; r < r1; ++r ) {
		const float iv = inv[r], c3 = iv * iv * iv * srow[r] / hidden, cr = crow[r];
		const size_t xb = static_cast<size_t>( r ) * kHc * hidden + d;
		float p[kHc], xh[kHc], u = 0.0f;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) {
			p[h] = pre[r * kHc + h];
			xh[h] = X[xb + static_cast<size_t>( h ) * hidden];
			u += p[h] * xh[h];
		}
		const float g = da[static_cast<size_t>( r ) * hidden + d];
		const float du = iv * g * wd - u * c3;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) dX[xb + static_cast<size_t>( h ) * hidden] += p[h] * du + cr * xh[h];
		acc += g * u * iv;
	}
	if ( gnorm != nullptr ) atomicAdd( &gnorm[d], acc );
}

// Block per output: gbase[j] += sum_r dz[r, j] (blocks 0 .. kHcMix - 1), gscale[t] += sum_r sum_{j of t} dz[r, j] l[r, j] (the next three)
__global__ void k_pre_bwd_bias( const float* dz, const float* mix, const float* rmr, int rows, float* gbase, float* gscale ) {
	const int o = blockIdx.x;
	if ( o < kHcMix ? gbase == nullptr : gscale == nullptr ) return;
	float s[1] = { 0.0f };
	if ( o < kHcMix ) {
		for ( int r = threadIdx.x; r < rows; r += kThreads ) s[0] += dz[static_cast<size_t>( r ) * kHcMix + o];
	}
	else {
		const int t = o - kHcMix, j0 = t == 0 ? 0 : t == 1 ? kHc : 2 * kHc, j1 = t == 0 ? kHc : t == 1 ? 2 * kHc : kHcMix;
		for ( int r = threadIdx.x; r < rows; r += kThreads ) {
			const float rr = rmr[r];
			for ( int j = j0; j < j1; ++j ) s[0] += dz[static_cast<size_t>( r ) * kHcMix + j] * mix[static_cast<size_t>( r ) * kHcMix + j] * rr;
		}
	}
	block_sums( s );
	if ( threadIdx.x == 0 ) {
		if ( o < kHcMix ) gbase[o] += s[0];
		else gscale[o - kHcMix] += s[0];
	}
}

// ---- post step ----

__global__ void k_post_fwd( const float* X_old, const float* post, const float* comb, const float* y, int rows, int hidden, float* X_new ) {
	MHC_LOOP( i, (size_t) rows * hidden ) {
		const size_t r = i / hidden, d = i % hidden, xb = r * kHc * hidden + d;
		float xo[kHc];
#pragma unroll
		for ( int g = 0; g < kHc; ++g ) xo[g] = X_old[xb + static_cast<size_t>( g ) * hidden];
		const float yv = y[i];
		const float* cb = comb + r * kHcComb;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) {
			float o = post[r * kHc + h] * yv;
#pragma unroll
			for ( int g = 0; g < kHc; ++g ) o += cb[g * kHc + h] * xo[g];
			X_new[xb + static_cast<size_t>( h ) * hidden] = o;
		}
	}
}

// dy = sum_h post_h dX_new_h, dX_old_g += sum_h comb[g][h] dX_new_h, per column
__global__ void k_post_bwd_cols( const float* post, const float* comb, const float* dX_new, int rows, int hidden, float* dX_old, float* dy ) {
	MHC_LOOP( i, (size_t) rows * hidden ) {
		const size_t r = i / hidden, d = i % hidden, xb = r * kHc * hidden + d;
		float dn[kHc], s = 0.0f;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) {
			dn[h] = dX_new[xb + static_cast<size_t>( h ) * hidden];
			s += post[r * kHc + h] * dn[h];
		}
		dy[i] = s;
		const float* cb = comb + r * kHcComb;
#pragma unroll
		for ( int g = 0; g < kHc; ++g ) {
			float o = 0.0f;
#pragma unroll
			for ( int h = 0; h < kHc; ++h ) o += cb[g * kHc + h] * dn[h];
			dX_old[xb + static_cast<size_t>( g ) * hidden] += o;
		}
	}
}

// Block per row: dpost_h = dX_new_h . y, dcomb[g][h] = dX_new_h . X_old_g
__global__ void k_post_bwd_dots( const float* X_old, const float* y, const float* dX_new, int hidden, float* dpost, float* dcomb ) {
	const int r = blockIdx.x;
	const size_t xb = static_cast<size_t>( r ) * kHc * hidden;
	float v[kHc + kHcComb];
#pragma unroll
	for ( int i = 0; i < kHc + kHcComb; ++i ) v[i] = 0.0f;
	for ( int d = threadIdx.x; d < hidden; d += kThreads ) {
		float dn[kHc], xo[kHc];
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) {
			dn[h] = dX_new[xb + static_cast<size_t>( h ) * hidden + d];
			xo[h] = X_old[xb + static_cast<size_t>( h ) * hidden + d];
		}
		const float yv = y[static_cast<size_t>( r ) * hidden + d];
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) v[h] += dn[h] * yv;
#pragma unroll
		for ( int g = 0; g < kHc; ++g )
#pragma unroll
			for ( int h = 0; h < kHc; ++h ) v[kHc + g * kHc + h] += dn[h] * xo[g];
	}
	block_sums( v );
	if ( threadIdx.x == 0 ) {
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) dpost[r * kHc + h] = v[h];
#pragma unroll
		for ( int i = 0; i < kHcComb; ++i ) dcomb[r * kHcComb + i] = v[kHc + i];
	}
}

// ---- head ----

__global__ void k_hc_mean( const float* X, int rows, int hidden, float* out ) {
	MHC_LOOP( i, (size_t) rows * hidden ) {
		const size_t r = i / hidden, d = i % hidden, xb = r * kHc * hidden + d;
		float s = 0.0f;
#pragma unroll
		for ( int h = 0; h < kHc; ++h ) s += X[xb + static_cast<size_t>( h ) * hidden];
		out[i] = s * ( 1.0f / kHc );
	}
}

__global__ void k_hc_mean_bwd( const float* dh, int rows, int hidden, float* dX ) {
	MHC_LOOP( j, (size_t) rows * kHc * hidden ) {
		const size_t r = j / ( (size_t) kHc * hidden ), d = j % hidden;
		dX[j] += dh[r * hidden + d] * ( 1.0f / kHc );
	}
}

} // namespace

MhcCache mhc_cache( float* buf, int rows ) {
	MhcCache c;
	c.mix = buf;
	c.rmr = c.mix + static_cast<size_t>( rows ) * kHcMix;
	c.pre = c.rmr + rows;
	c.inv = c.pre + static_cast<size_t>( rows ) * kHc;
	return c;
}

void mhc_pre_forward( Blas& b, const MhcSite& s, const float* X, int rows, const MhcCache& c, float* a, float* post, float* comb ) {
	ENGINE_ASSERT( s.hidden > 0 && s.iters >= 1, "mhc_pre_forward: bad site" );
	if ( rows <= 0 ) return;
	const cudaStream_t st = cu( b.stream() );
	const int n = kHc * s.hidden;
	b.gemm( false, true, rows, kHcMix, n, 1.0f, X, s.fn, 0.0f, c.mix );
	k_pre_gates<<<rows, kThreads, 0, st>>>( X, c.mix, s.base, s.scale, s.hidden, s.rms_eps, s.hc_eps, c.rmr, c.pre, post );
	k_pre_comb<<<( rows + kRowThreads - 1 ) / kRowThreads, kRowThreads, 0, st>>>( c.mix, c.rmr, s.base, s.scale, rows, s.iters, s.hc_eps, comb );
	k_pre_collapse<<<rows, kThreads, 0, st>>>( X, c.pre, s.norm_w, s.hidden, s.norm_eps, c.inv, a );
}

size_t mhc_pre_scratch_floats( const MhcSite& s, int rows ) {
	return static_cast<size_t>( rows ) * ( 2 + kHc + 2 * kHcMix + static_cast<size_t>( 2 * s.iters - 1 ) * kHcComb );
}

void mhc_pre_backward( Blas& b, const MhcSite& s, const float* X, int rows, const MhcCache& c, const float* da, const float* dpost, const float* dcomb,
	float* dX, float* scratch ) {
	ENGINE_ASSERT( s.hidden > 0 && s.iters >= 1 && da != nullptr, "mhc_pre_backward: bad site or no da" );
	if ( rows <= 0 ) return;
	const cudaStream_t st = cu( b.stream() );
	const int n = kHc * s.hidden;
	float* srow = scratch;
	float* crow = srow + rows;
	float* dpre = crow + rows;
	float* dz = dpre + static_cast<size_t>( rows ) * kHc;
	float* dmix = dz + static_cast<size_t>( rows ) * kHcMix;
	float* iter = dmix + static_cast<size_t>( rows ) * kHcMix;
	k_pre_bwd_row<<<rows, kThreads, 0, st>>>( X, da, s.norm_w, c.pre, c.inv, s.hidden, srow, dpre );
	k_pre_bwd_gates<<<( rows + kRowThreads - 1 ) / kRowThreads, kRowThreads, 0, st>>>( c.mix, c.rmr, s.base, s.scale, dpre, dpost, dcomb, rows, s.hidden,
		s.iters, s.hc_eps, dz, dmix, crow, iter );
	b.gemm( false, false, rows, n, kHcMix, 1.0f, dmix, s.fn, 1.0f, dX );                // mix = X fn^T
	if ( s.gfn ) b.gemm( true, false, kHcMix, n, rows, 1.0f, dmix, X, 1.0f, s.gfn );
	const int chunks = std::max( 1, std::min( 64, rows / 64 ) ), per_chunk = ( rows + chunks - 1 ) / chunks;
	k_pre_bwd_dx<<<dim3( ( s.hidden + kThreads - 1 ) / kThreads, chunks ), kThreads, 0, st>>>( X, da, s.norm_w, c.pre, c.inv, srow, crow, rows, s.hidden,
		per_chunk, dX, s.gnorm );
	if ( s.gbase || s.gscale ) k_pre_bwd_bias<<<kHcMix + 3, kThreads, 0, st>>>( dz, c.mix, c.rmr, rows, s.gbase, s.gscale );
}

void mhc_post_forward( const float* X_old, const float* post, const float* comb, const float* y, int rows, int hidden, float* X_new,
	backend::stream_t stream ) {
	if ( rows <= 0 ) return;
	k_post_fwd<<<blocks_for( (size_t) rows * hidden ), 256, 0, cu( stream )>>>( X_old, post, comb, y, rows, hidden, X_new );
}

void mhc_post_backward( const float* X_old, const float* post, const float* comb, const float* y, const float* dX_new, int rows, int hidden, float* dX_old,
	float* dy, float* dpost, float* dcomb, backend::stream_t stream ) {
	if ( rows <= 0 ) return;
	k_post_bwd_dots<<<rows, kThreads, 0, cu( stream )>>>( X_old, y, dX_new, hidden, dpost, dcomb );
	k_post_bwd_cols<<<blocks_for( (size_t) rows * hidden ), 256, 0, cu( stream )>>>( post, comb, dX_new, rows, hidden, dX_old, dy );
}

void hc_mean_forward( const float* X, int rows, int hidden, float* h, backend::stream_t stream ) {
	if ( rows <= 0 ) return;
	k_hc_mean<<<blocks_for( (size_t) rows * hidden ), 256, 0, cu( stream )>>>( X, rows, hidden, h );
}

void hc_mean_backward( const float* dh, int rows, int hidden, float* dX, backend::stream_t stream ) {
	if ( rows <= 0 ) return;
	k_hc_mean_bwd<<<blocks_for( (size_t) rows * kHc * hidden ), 256, 0, cu( stream )>>>( dh, rows, hidden, dX );
}

} // namespace engine::train::glm5_next
