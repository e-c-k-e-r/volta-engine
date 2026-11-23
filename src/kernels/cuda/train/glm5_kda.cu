// Training forms of GLM-5.3-Flash operations: kda (engine/train/glm5_kda.h). The recurrence follows qwen4_exp's GDN kernels (the same blocks,
// tiles and checkpoint scheme), the scalar gate replaced by a decay per key channel: dgate becomes a vector per row like dq and dk.
#include "engine/train/glm5_kda.h"
#include "engine/config.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace engine::train::glm5_next {

namespace {

#define KDA_LOOP( i, n ) for ( size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < ( n ); i += (size_t) gridDim.x * blockDim.x )

unsigned blocks_for( size_t n ) { return static_cast<unsigned>( ( n + 255 ) / 256 < 65535u * 8u ? ( n + 255 ) / 256 : 65535u * 8u ); }

cudaStream_t cu( backend::stream_t s ) { return static_cast<cudaStream_t>( s ); }

__device__ __forceinline__ float sigmoidf_( float x ) { return 1.0f / ( 1.0f + expf( -x ) ); }

__device__ __forceinline__ float warp_sum( float v ) {
	for ( int o = 16; o > 0; o >>= 1 ) v += __shfl_xor_sync( 0xffffffffu, v, o );
	return v;
}

// ---- l2 norm: a warp per segment, 8 a block ----
__global__ void k_l2_fwd( const float* x, int segments, int n, float eps, float scale, float* inv, float* y ) {
	const int seg = blockIdx.x * 8 + ( threadIdx.x >> 5 ), lane = threadIdx.x & 31;
	if ( seg >= segments ) return;
	const size_t base = static_cast<size_t>( seg ) * n;
	float s = 0.0f;
	for ( int i = lane; i < n; i += 32 ) s = fmaf( x[base + i], x[base + i], s );
	s = warp_sum( s );
	const float iv = rsqrtf( s + eps ), m = iv * scale;
	if ( lane == 0 ) inv[seg] = iv;
	for ( int i = lane; i < n; i += 32 ) y[base + i] = x[base + i] * m;
}

// dx += scale ( inv dy - x inv^3 ( dy . x ) )
__global__ void k_l2_bwd( const float* x, int segments, int n, const float* inv, float scale, const float* dy, float* dx ) {
	const int seg = blockIdx.x * 8 + ( threadIdx.x >> 5 ), lane = threadIdx.x & 31;
	if ( seg >= segments ) return;
	const size_t base = static_cast<size_t>( seg ) * n;
	float s = 0.0f;
	for ( int i = lane; i < n; i += 32 ) s = fmaf( dy[base + i], x[base + i], s );
	s = warp_sum( s );
	const float iv = inv[seg], c = iv * iv * iv * s;
	for ( int i = lane; i < n; i += 32 ) dx[base + i] += scale * ( iv * dy[base + i] - x[base + i] * c );
}

// ---- gates ----
__global__ void k_kda_gates( const float* b, const float* fpre, const float* dt_bias, const float* A_log, int rows, int heads, int D, float lower, float* beta,
	float* g ) {
	const size_t cols = static_cast<size_t>( heads ) * D;
	KDA_LOOP( i, rows * cols ) {
		const int j = static_cast<int>( i % cols );
		g[i] = lower * sigmoidf_( expf( A_log[j / D] ) * ( fpre[i] + dt_bias[j] ) );
	}
	KDA_LOOP( i, (size_t) rows * heads ) beta[i] = sigmoidf_( b[i] );
}

// Thread per channel j over blockIdx.y's span of kGateRows rows: dfpre = dg lower s ( 1 - s ) e^A_log (s the sigmoid), its sums over the span
// into d_dt_bias[j] and, times f, into dA_log[h] (a warp's sum when D is a multiple of 32: one head per warp); the first heads threads write db
constexpr int kGateRows = 64;
__global__ void k_kda_gates_bwd( const float* b, const float* fpre, const float* dt_bias, const float* A_log, int rows, int heads, int D, float lower,
	const float* dbeta, const float* dg, float* db, float* dfpre, float* d_dt_bias, float* dA_log ) {
	const int cols = heads * D, j = blockIdx.x * blockDim.x + threadIdx.x;
	const int r0 = blockIdx.y * kGateRows, r1 = min( rows, r0 + kGateRows );
	if ( j < heads ) {
		for ( int r = r0; r < r1; ++r ) {
			const size_t o = static_cast<size_t>( r ) * heads + j;
			const float s = sigmoidf_( b[o] );
			db[o] = dbeta[o] * s * ( 1.0f - s );
		}
	}
	float adt = 0.0f, aA = 0.0f;
	if ( j < cols ) {
		const float e = expf( A_log[j / D] ), bias = dt_bias[j];
		for ( int r = r0; r < r1; ++r ) {
			const size_t o = static_cast<size_t>( r ) * cols + j;
			const float f = fpre[o] + bias, s = sigmoidf_( e * f ), df = dg[o] * lower * s * ( 1.0f - s ) * e;
			dfpre[o] = df;
			adt += df;
			aA = fmaf( df, f, aA );
		}
		if ( d_dt_bias ) atomicAdd( &d_dt_bias[j], adt );
	}
	if ( dA_log == nullptr ) return;
	if ( D % 32 == 0 ) {
		// cols is a multiple of 32: a warp lies wholly inside or outside the channels
		aA = warp_sum( aA );
		if ( ( threadIdx.x & 31 ) == 0 && j < cols ) atomicAdd( &dA_log[j / D], aA );
	}
	else if ( j < cols ) atomicAdd( &dA_log[j / D], aA );
}

// ---- the delta rule ----
constexpr int KD = kKdaDim;   // head dim
constexpr int KW = 16;        // value columns per block, a half-warp each

// The forward's row tiles: kKdaTile rows of k, q, e^g, beta, k . q and the block's 16 value columns, staged in shared memory, the next tile's loads
// in flight while a tile is used
constexpr int kKdaTile = 16;

// Block (head, 16 columns, sequence) of 8 warps, a half-warp per value column c holding its state column S[., c] in 8 values per lane (key channel
// r * 16 + l); the rows hold gridDim.z sequences of rows / gridDim.z, each from a zero state. With S' = diag( e^g ) S the decayed state,
// o_t[c] = S'_c . q_t + d_c ( k_t . q_t ), so both sums of a row run together. The decay is a plain rounded product (__fmul_rn) and the update an
// explicit fma, as in the backward's recompute: the recomputed states are the forward's bit for bit
__global__ __launch_bounds__( 256 )
void k_kda_fwd( const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ g,
	const float* __restrict__ beta, int rows, int heads, int chunk, float* __restrict__ ckpt, float* __restrict__ o ) {
	__shared__ float4 sk4[kKdaTile * KD / 4], sq4[kKdaTile * KD / 4], se4[kKdaTile * KD / 4];
	__shared__ float sb[kKdaTile], skq[kKdaTile], sv[kKdaTile][KW];
	const int h = blockIdx.x, c0 = blockIdx.y * KW, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l = lane & 15;
	const int cl = w * 2 + ( lane >> 4 ), c = c0 + cl;
	const int L = rows / gridDim.z, nch = ( L + chunk - 1 ) / chunk, s0 = blockIdx.z * L, s1 = s0 + L;
	const float* sk = reinterpret_cast<const float*>( sk4 );
	const float* sq = reinterpret_cast<const float*>( sq4 );
	const float* se = reinterpret_cast<const float*>( se4 );
	// this thread's share of a tile: row tid / 16 of k, q and g (8 values each: two float4 at part tid % 16), one v value, one beta
	const int lr = tid >> 4, lp = tid & 15;
	float4 pk[2], pq[2], pe[2];
	float pv = 0.0f, pb = 0.0f;
	auto fetch = [&]( int t0 ) {
		const int t = t0 + lr;
		if ( t < s1 ) {
			const size_t base = ( static_cast<size_t>( t ) * heads + h ) * KD;
			const float4* kt = reinterpret_cast<const float4*>( k + base ) + lp * 2;
			const float4* qt = reinterpret_cast<const float4*>( q + base ) + lp * 2;
			const float4* gt = reinterpret_cast<const float4*>( g + base ) + lp * 2;
			pk[0] = kt[0]; pk[1] = kt[1]; pq[0] = qt[0]; pq[1] = qt[1];
			for ( int x = 0; x < 2; ++x ) {
				const float4 a = gt[x];
				pe[x] = make_float4( expf( a.x ), expf( a.y ), expf( a.z ), expf( a.w ) );
			}
			pv = v[base + c0 + lp];
		}
		else {
			pk[0] = pk[1] = pq[0] = pq[1] = pe[0] = pe[1] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
			pv = 0.0f;
		}
		const int tb = t0 + tid;
		if ( tid < kKdaTile ) pb = tb < s1 ? beta[static_cast<size_t>( tb ) * heads + h] : 0.0f;
	};
	float S[8];
	for ( int r = 0; r < 8; ++r ) S[r] = 0.0f;
	fetch( s0 );
	for ( int t0 = s0; t0 < s1; t0 += kKdaTile ) {
		__syncthreads();   // the previous tile used
		const int p = lr * ( KD / 4 ) + lp * 2;
		sk4[p] = pk[0]; sk4[p + 1] = pk[1];
		sq4[p] = pq[0]; sq4[p + 1] = pq[1];
		se4[p] = pe[0]; se4[p + 1] = pe[1];
		sv[lr][lp] = pv;
		if ( tid < kKdaTile ) sb[tid] = pb;
		__syncthreads();
		{
			// k . q of the tile's rows: row w * 2 + half-warp
			const int i = w * 2 + ( lane >> 4 );
			float kq = 0.0f;
			for ( int r = 0; r < 8; ++r ) kq = fmaf( sk[i * KD + r * 16 + l], sq[i * KD + r * 16 + l], kq );
			for ( int m = 8; m > 0; m >>= 1 ) kq += __shfl_xor_sync( 0xffffffffu, kq, m );
			if ( l == 0 ) skq[i] = kq;
		}
		if ( t0 + kKdaTile < s1 ) fetch( t0 + kKdaTile );
		__syncthreads();
		const int n = min( kKdaTile, s1 - t0 );
		for ( int i = 0; i < n; ++i ) {
			const int t = t0 + i;
			if ( ( t - s0 ) % chunk == 0 ) {
				float* ck = ckpt + ( ( ( static_cast<size_t>( blockIdx.z ) * nch + ( t - s0 ) / chunk ) * heads + h ) * KD + c ) * KD;
				for ( int r = 0; r < 8; ++r ) ck[r * 16 + l] = S[r];
			}
			float kk[8], u = 0.0f, sqv = 0.0f;
			for ( int r = 0; r < 8; ++r ) {
				const int x = i * KD + r * 16 + l;
				S[r] = __fmul_rn( S[r], se[x] );
				kk[r] = sk[x];
				u = fmaf( S[r], kk[r], u );
				sqv = fmaf( S[r], sq[x], sqv );
			}
			for ( int m = 8; m > 0; m >>= 1 ) {
				u += __shfl_xor_sync( 0xffffffffu, u, m );
				sqv += __shfl_xor_sync( 0xffffffffu, sqv, m );
			}
			const float d = sb[i] * ( sv[i][cl] - u );
			for ( int r = 0; r < 8; ++r ) S[r] = fmaf( d, kk[r], S[r] );
			if ( l == 0 ) o[( static_cast<size_t>( t ) * heads + h ) * KD + c] = fmaf( d, skq[i], sqv );
		}
	}
}

// The backward's sub-chunks: kKdaSub rows whose states a thread holds in registers at once
constexpr int kKdaSub = 8;

// The backward, block (head, 16 columns, sequence) of 8 warps in the forward's layout, the chunks last first and dS (the gradient of the state
// after the current row) in registers throughout. Per chunk: from its checkpoint the states at its sub-chunks' starts (into sub, a column's
// kKdaSub-row sub-checkpoints); then per sub-chunk, last first, its states recomputed into registers and its rows in reverse. Per row and column,
// with S' = e^g S_{t-1}, S_t = S' + d k and dd = dS . k after dS += dout q:
//   dq += dout S_t, dv = beta dd, dbeta += dd ( v - m ), dm = -beta dd, dS' = dS + dm k, dk += d dS + dm S', dg += dS' S', dS <- e^g dS'
// The sums over the block's columns (dq, dk, dg: a vector each; dbeta) go through per-warp partials in shared memory (two buffers, one barrier a
// row) and one global add per entry
__global__ __launch_bounds__( 256, 2 )
void k_kda_bwd( const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ g,
	const float* __restrict__ beta, int rows, int heads, int chunk, const float* __restrict__ ckpt, const float* __restrict__ dout, float* dq, float* dk,
	float* dv, float* dg, float* dbeta, float* __restrict__ sub ) {
	__shared__ float4 sk4[kKdaSub * KD / 4], sq4[kKdaSub * KD / 4], se4[kKdaSub * KD / 4];
	__shared__ float sb[kKdaSub], sv[kKdaSub][KW], sdo[kKdaSub][KW], su[kKdaSub][KW], sd[kKdaSub][KW];
	__shared__ float sp[2][8][3 * KD], sc[2][KW];   // per warp a row's dq | dk | dg partials; per column its dbeta term
	const int h = blockIdx.x, c0 = blockIdx.y * KW, tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l = lane & 15;
	const int cl = w * 2 + ( lane >> 4 ), c = c0 + cl;
	const int L = rows / gridDim.z, nch = ( L + chunk - 1 ) / chunk, s0 = blockIdx.z * L, s1 = s0 + L, nsub = ( chunk + kKdaSub - 1 ) / kKdaSub;
	const float* skf = reinterpret_cast<const float*>( sk4 );
	const float* sqf = reinterpret_cast<const float*>( sq4 );
	const float* sef = reinterpret_cast<const float*>( se4 );
	float* subc = sub + ( ( static_cast<size_t>( blockIdx.z ) * heads + h ) * KD + c ) * nsub * KD;
	int buf = 0;
	// rows [t0, t0 + n) into the tiles: k, e^g, beta, v, with all also q and dout
	auto stage = [&]( int t0, int n, bool all ) {
		__syncthreads();
		const int r = tid >> 5, p = tid & 31;
		const size_t kb = ( static_cast<size_t>( t0 + r ) * heads + h ) * KD;
		const float4 z4 = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
		sk4[tid] = r < n ? reinterpret_cast<const float4*>( k + kb )[p] : z4;
		if ( r < n ) {
			const float4 a = reinterpret_cast<const float4*>( g + kb )[p];
			se4[tid] = make_float4( expf( a.x ), expf( a.y ), expf( a.z ), expf( a.w ) );
		}
		else se4[tid] = z4;
		if ( all ) sq4[tid] = r < n ? reinterpret_cast<const float4*>( q + kb )[p] : z4;
		const int rv = ( tid & 127 ) >> 4, j = tid & 15;
		const size_t vb = ( static_cast<size_t>( t0 + rv ) * heads + h ) * KD + c0 + j;
		if ( tid < 128 ) sv[rv][j] = rv < n ? v[vb] : 0.0f;
		else if ( all ) sdo[rv][j] = rv < n ? dout[vb] : 0.0f;
		if ( tid < kKdaSub ) sb[tid] = tid < n ? beta[static_cast<size_t>( t0 + tid ) * heads + h] : 0.0f;
		__syncthreads();
	};
	float dS[8];
	for ( int r = 0; r < 8; ++r ) dS[r] = 0.0f;
	for ( int n = nch - 1; n >= 0; --n ) {
		const int cs = s0 + n * chunk, ce = min( s1, cs + chunk ), ns = ( ce - cs + kKdaSub - 1 ) / kKdaSub;
		const float* ck = ckpt + ( ( ( static_cast<size_t>( blockIdx.z ) * nch + n ) * heads + h ) * KD + c ) * KD;
		float S[8];
		for ( int r = 0; r < 8; ++r ) S[r] = ck[r * 16 + l];
		// the states at the starts of sub-chunks 1 .. ns - 1 (the last one stays in S)
		for ( int j = 0; j + 1 < ns; ++j ) {
			stage( cs + j * kKdaSub, kKdaSub, false );
			for ( int i = 0; i < kKdaSub; ++i ) {
				float kk[8], u = 0.0f;
				for ( int r = 0; r < 8; ++r ) {
					const int x = i * KD + r * 16 + l;
					S[r] = __fmul_rn( S[r], sef[x] );
					kk[r] = skf[x];
					u = fmaf( S[r], kk[r], u );
				}
				for ( int m = 8; m > 0; m >>= 1 ) u += __shfl_xor_sync( 0xffffffffu, u, m );
				const float d = sb[i] * ( sv[i][cl] - u );
				for ( int r = 0; r < 8; ++r ) S[r] = fmaf( d, kk[r], S[r] );
			}
			if ( j + 2 < ns ) for ( int r = 0; r < 8; ++r ) subc[( j + 1 ) * KD + r * 16 + l] = S[r];
		}
		for ( int j = ns - 1; j >= 0; --j ) {
			const int t0 = cs + j * kKdaSub, nr = min( kKdaSub, ce - t0 );
			if ( j < ns - 1 ) {
				const float* from = j == 0 ? ck : subc + j * KD;
				for ( int r = 0; r < 8; ++r ) S[r] = from[r * 16 + l];
			}
			stage( t0, nr, true );
			// the sub-chunk's decayed states S' of each row (Sx), m and d per row and column
			float Sx[kKdaSub][8];
#pragma unroll
			for ( int i = 0; i < kKdaSub; ++i ) {
				if ( i < nr ) {
					float kk[8], u = 0.0f;
					for ( int r = 0; r < 8; ++r ) {
						const int x = i * KD + r * 16 + l;
						S[r] = __fmul_rn( S[r], sef[x] );
						Sx[i][r] = S[r];
						kk[r] = skf[x];
						u = fmaf( S[r], kk[r], u );
					}
					for ( int m = 8; m > 0; m >>= 1 ) u += __shfl_xor_sync( 0xffffffffu, u, m );
					const float d = sb[i] * ( sv[i][cl] - u );
					if ( l == 0 ) {
						su[i][cl] = u;
						sd[i][cl] = d;
					}
					for ( int r = 0; r < 8; ++r ) S[r] = fmaf( d, kk[r], S[r] );
				}
				else for ( int r = 0; r < 8; ++r ) Sx[i][r] = 0.0f;
			}
			__syncwarp();
#pragma unroll
			for ( int i = kKdaSub - 1; i >= 0; --i ) {
				if ( i >= nr ) continue;
				const int t = t0 + i;
				const float bt = sb[i], u = su[i][cl], d = sd[i][cl], doc = sdo[i][cl];
				float dd = 0.0f;
				for ( int r = 0; r < 8; ++r ) {
					const int x = i * KD + r * 16 + l;
					const float kk = skf[x];
					const float a = doc * fmaf( d, kk, Sx[i][r] );   // dout S_t
					const float a2 = a + __shfl_xor_sync( 0xffffffffu, a, 16 );
					if ( lane < 16 ) sp[buf][w][r * 16 + l] = a2;
					dS[r] = fmaf( doc, sqf[x], dS[r] );
					dd = fmaf( dS[r], kk, dd );
				}
				for ( int m = 8; m > 0; m >>= 1 ) dd += __shfl_xor_sync( 0xffffffffu, dd, m );
				const float dm = -bt * dd;
				for ( int r = 0; r < 8; ++r ) {
					const int x = i * KD + r * 16 + l;
					const float kk = skf[x], Sp = Sx[i][r];
					const float dSp = fmaf( dm, kk, dS[r] );
					float dkc = fmaf( dS[r], d, dm * Sp ), dgc = dSp * Sp;
					dS[r] = sef[x] * dSp;
					dkc += __shfl_xor_sync( 0xffffffffu, dkc, 16 );
					dgc += __shfl_xor_sync( 0xffffffffu, dgc, 16 );
					if ( lane < 16 ) {
						sp[buf][w][KD + r * 16 + l] = dkc;
						sp[buf][w][2 * KD + r * 16 + l] = dgc;
					}
				}
				if ( l == 0 ) {
					dv[( static_cast<size_t>( t ) * heads + h ) * KD + c] += bt * dd;
					sc[buf][cl] = dd * ( sv[i][cl] - u );
				}
				__syncthreads();
				const size_t rb = ( static_cast<size_t>( t ) * heads + h ) * KD;
				float s = 0.0f;
				for ( int x = 0; x < 8; ++x ) s += sp[buf][x][tid];
				atomicAdd( ( tid < KD ? dq : dk ) + rb + ( tid & ( KD - 1 ) ), s );
				if ( tid < KD ) {
					float s2 = 0.0f;
					for ( int x = 0; x < 8; ++x ) s2 += sp[buf][x][2 * KD + tid];
					atomicAdd( dg + rb + tid, s2 );
				}
				else if ( tid == KD ) {
					float z = 0.0f;
					for ( int x = 0; x < KW; ++x ) z += sc[buf][x];
					atomicAdd( dbeta + static_cast<size_t>( t ) * heads + h, z );
				}
				buf ^= 1;
			}
		}
	}
}

bool aligned16( const void* p ) { return ( reinterpret_cast<uintptr_t>( p ) & 15u ) == 0; }

} // namespace

void l2norm_forward( const float* x, int segments, int n, float eps, float scale, float* inv, float* y, backend::stream_t stream ) {
	if ( segments > 0 ) k_l2_fwd<<<( segments + 7 ) / 8, 256, 0, cu( stream )>>>( x, segments, n, eps, scale, inv, y );
}

void l2norm_backward( const float* x, int segments, int n, const float* inv, float scale, const float* dy, float* dx, backend::stream_t stream ) {
	if ( segments > 0 ) k_l2_bwd<<<( segments + 7 ) / 8, 256, 0, cu( stream )>>>( x, segments, n, inv, scale, dy, dx );
}

void kda_gates_forward( const float* b, const float* fpre, const float* dt_bias, const float* A_log, int rows, int heads, int D, float lower, float* beta,
	float* g, backend::stream_t stream ) {
	if ( rows > 0 ) k_kda_gates<<<blocks_for( (size_t) rows * heads * D ), 256, 0, cu( stream )>>>( b, fpre, dt_bias, A_log, rows, heads, D, lower, beta, g );
}

void kda_gates_backward( const float* b, const float* fpre, const float* dt_bias, const float* A_log, int rows, int heads, int D, float lower, const float* dbeta,
	const float* dg, float* db, float* dfpre, float* d_dt_bias, float* dA_log, backend::stream_t stream ) {
	ENGINE_ASSERT( heads > 0 && D > 0, "kda_gates_backward: heads and D must be positive" );
	if ( rows > 0 )
		k_kda_gates_bwd<<<dim3( ( heads * D + 255 ) / 256, ( rows + kGateRows - 1 ) / kGateRows ), 256, 0, cu( stream )>>>( b, fpre, dt_bias, A_log, rows,
			heads, D, lower, dbeta, dg, db, dfpre, d_dt_bias, dA_log );
}

size_t kda_ckpt_floats( int rows, int heads, int chunk, int seqs ) {
	return static_cast<size_t>( seqs ) * ( ( rows / seqs + chunk - 1 ) / chunk ) * heads * KD * KD;
}

size_t kda_scratch_floats( int heads, int chunk, int seqs ) {
	return static_cast<size_t>( seqs ) * heads * KD * KD * ( ( chunk + kKdaSub - 1 ) / kKdaSub );
}

void kda_forward( const float* q, const float* k, const float* v, const float* g, const float* beta, int rows, int heads, int chunk, float* ckpt, float* o,
	backend::stream_t stream, int seqs ) {
	ENGINE_ASSERT( seqs > 0 && rows % seqs == 0, "kda_forward: the rows must hold whole sequences" );
	ENGINE_ASSERT( chunk > 0, "kda_forward: chunk must be positive" );
	ENGINE_ASSERT( aligned16( q ) && aligned16( k ) && aligned16( g ), "kda_forward: q, k and g must be 16-byte aligned" );
	if ( rows > 0 ) k_kda_fwd<<<dim3( heads, KD / KW, seqs ), 256, 0, cu( stream )>>>( q, k, v, g, beta, rows, heads, chunk, ckpt, o );
}

void kda_backward( const float* q, const float* k, const float* v, const float* g, const float* beta, int rows, int heads, int chunk, const float* ckpt,
	const float* dout, float* dq, float* dk, float* dv, float* dg, float* dbeta, float* scratch, backend::stream_t stream, int seqs ) {
	ENGINE_ASSERT( seqs > 0 && rows % seqs == 0, "kda_backward: the rows must hold whole sequences" );
	ENGINE_ASSERT( chunk > 0, "kda_backward: chunk must be positive" );
	ENGINE_ASSERT( aligned16( q ) && aligned16( k ) && aligned16( g ), "kda_backward: q, k and g must be 16-byte aligned" );
	if ( rows > 0 )
		k_kda_bwd<<<dim3( heads, KD / KW, seqs ), 256, 0, cu( stream )>>>( q, k, v, g, beta, rows, heads, chunk, ckpt, dout, dq, dk, dv, dg, dbeta, scratch );
}

} // namespace engine::train::glm5_next
