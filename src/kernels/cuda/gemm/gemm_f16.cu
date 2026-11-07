// Dense FP16 GEMM on Volta's tensor cores (mma.sync.m8n8k4, FP32 accumulation): out[m, n] = beta * out + a[m, k] * w[n, k]^T (tn) or
// a[m, k] * w[k, n] (nn). A block computes a BM x BN tile over k in steps of BK: both operands' tiles are staged in shared memory (16-byte
// chunks XOR-swizzled by row, so fragment reads and tile stores hit every bank once), two stages: the step after next loads into registers
// right after the barrier, fragments are double-buffered across the mma k steps. Warps tile the block 2 x 2 (64 x 64 each at 128 x 128);
// quad pairs 0 / 1 and 2 / 3 split a 16 x 16 fragment's rows and columns (frag_line). The tile size follows the row count, a split along k
// fills the GPU when the tiles do not (split_reduce adds the partials in order), and the epilogue goes through shared memory to 16-byte stores.
// Fragment layouts of m8n8k4 (measured on sm_70): a lane holds row p of A and column p of B (p = lane % 4 + 4 * (lane >= 16)), 4 consecutive
// k each (B row-major: k row lane % 4, columns 4 (lane >= 16) + 0..3); its 8 accumulators are rows (lane & 1) + 4 (lane >= 16) and + 2,
// columns (lane & 2) + {0, 1, 4, 5}, in the order (r, c) (r, c + 1) (r + 2, c) (r + 2, c + 1) and the same 4 columns on.
#include "engine/modules/gemm.h"
#include "engine/config.h"
#include <cuda_fp16.h>
#include <type_traits>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>

#if defined(USE_ROCM)
#include <cublas_v2.h>   // hipBLAS through the ROCm stand-in
#endif

#if !defined(USE_ROCM)
namespace engine::modules {

namespace {


// B row-major (k rows of 8 n): a lane holds k row lane % 4, columns 4 (lane >= 16) + 0..3
__device__ __forceinline__ void mma884_rr( float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1 ) {
	asm( "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
		: "+f"( d[0] ), "+f"( d[1] ), "+f"( d[2] ), "+f"( d[3] ), "+f"( d[4] ), "+f"( d[5] ), "+f"( d[6] ), "+f"( d[7] ) : "r"( a0 ), "r"( a1 ), "r"( b0 ), "r"( b1 ) );
}

__device__ __forceinline__ void mma884( float d[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1 ) {
	asm( "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
		: "+f"( d[0] ), "+f"( d[1] ), "+f"( d[2] ), "+f"( d[3] ), "+f"( d[4] ), "+f"( d[5] ), "+f"( d[6] ), "+f"( d[7] ) : "r"( a0 ), "r"( a1 ), "r"( b0 ), "r"( b1 ) );
}

// One operand's k step: rows [r0, r0 + ROWS) of a row-major [rows, k] matrix, values [k0, k0 + BK), as 16-byte chunks (zero past the edges)
template <int ROWS, int BK, int THREADS>
struct Stage {
	static constexpr int kBK = BK;
	// 16-byte chunk c of row r sits at chunk c ^ swizzle( r ): the 8 rows a quarter-warp reads (and the rows it stores) cover all 32 banks
	static __device__ __forceinline__ int swz( int r, int c ) { return c ^ ( ( r / ( 64 / BK ) ) % ( BK / 8 ) ); }
	static constexpr int chunks = ROWS * kBK / 8, per_thread = ( chunks + THREADS - 1 ) / THREADS;
	uint4 v[per_thread];
	__device__ __forceinline__ void load( const half* __restrict__ p, int ld, int rows, int k, int r0, int k0 ) {
		#pragma unroll
		for ( int i = 0; i < per_thread; ++i ) {
			const int idx = threadIdx.x + i * THREADS, r = idx / ( kBK / 8 ), c = 8 * ( idx % ( kBK / 8 ) );
			if ( chunks % THREADS != 0 && idx >= chunks ) break;
			v[i] = r0 + r < rows && k0 + c < k ? __ldg( reinterpret_cast<const uint4*>( p + static_cast<size_t>( r0 + r ) * ld + k0 + c ) ) : make_uint4( 0, 0, 0, 0 );
		}
	}
	__device__ __forceinline__ void store( half* s ) const {
		#pragma unroll
		for ( int i = 0; i < per_thread; ++i ) {
			const int idx = threadIdx.x + i * THREADS, r = idx / ( kBK / 8 ), c = 8 * ( idx % ( kBK / 8 ) );
			if ( chunks % THREADS != 0 && idx >= chunks ) break;
			*reinterpret_cast<uint4*>( s + r * kBK + 8 * swz( r, c / 8 ) ) = v[i];
		}
	}
};

// The weights' k step when they are stored [k, n] (n contiguous): rows [k0, k0 + BK), columns [n0, n0 + BN) as 16-byte chunks (zero past the
// edges, n a multiple of 8); 16-byte chunk c of k row r sits at chunk c ^ ( r % 4 ), so the 4 k rows a quad reads fall in different banks
template <int BK, int BN, int THREADS>
struct StageKN {
	static constexpr int chunks = BK * BN / 8, per_thread = ( chunks + THREADS - 1 ) / THREADS;
	uint4 v[per_thread];
	__device__ __forceinline__ void load( const half* __restrict__ p, int ld, int n, int k, int n0, int k0 ) {
		#pragma unroll
		for ( int i = 0; i < per_thread; ++i ) {
			const int idx = threadIdx.x + i * THREADS, r = idx / ( BN / 8 ), c = 8 * ( idx % ( BN / 8 ) );
			if ( chunks % THREADS != 0 && idx >= chunks ) break;
			v[i] = k0 + r < k && n0 + c < n ? __ldg( reinterpret_cast<const uint4*>( p + static_cast<size_t>( k0 + r ) * ld + n0 + c ) ) : make_uint4( 0, 0, 0, 0 );
		}
	}
	__device__ __forceinline__ void store( half* s ) const {
		#pragma unroll
		for ( int i = 0; i < per_thread; ++i ) {
			const int idx = threadIdx.x + i * THREADS, r = idx / ( BN / 8 ), c = 8 * ( idx % ( BN / 8 ) );
			if ( chunks % THREADS != 0 && idx >= chunks ) break;
			*reinterpret_cast<uint4*>( s + r * BN + 8 * ( ( c / 8 ) ^ ( r & 3 ) ) ) = v[i];
		}
	}
};

template <bool WKN> constexpr int w_stage_halves( int BN, int BK ) { return BN * BK; }

// Row (of A) or column (of W) within a 16-wide fragment that a quad pair's row / column r of its 8 x 8 product maps to: quad pairs 0 and 1
// (2 and 3) split the fragment's rows (columns) as 0-3 + 8-11 and 4-7 + 12-15, so each half-warp reads 8 consecutive rows (no bank conflicts)
__device__ __forceinline__ int frag_line( int half_id, int r ) { return ( r & 3 ) + 4 * half_id + 8 * ( r >> 2 ); }

// Split along k (gridDim.z > 1): each split's partial sums go to part[z][m][n] (FP32), the reduce adds them in split order
// WKN: the weights are stored [k, n] (out = a * w) instead of [n, k] (out = a * w^T)
template <int BM, int BN, int BK, int WM, int WN, bool F32, bool WKN>
__global__ __launch_bounds__( WM * WN * 32 )
void gemm_tn_kernel( const half* __restrict__ a, int lda, const half* __restrict__ w, int ldw, void* __restrict__ out, int ldc, int m, int n, int k, float beta,
	float* __restrict__ part, int k_per_split ) {
	constexpr int THREADS = WM * WN * 32, TM = BM / WM, TN = BN / WN, MB = TM / 16, NB = TN / 16;
	constexpr int kBK = BK;
	extern __shared__ __align__( 16 ) half smem[];
	half( *as )[BM][BK] = reinterpret_cast<half( * )[BM][BK]>( smem );
	half* ws = smem + 2 * BM * BK;
	constexpr int WSTAGE = w_stage_halves<WKN>( BN, BK );
	using SA = Stage<BM, BK, WM * WN * 32>;
	using SW = std::conditional_t<WKN, StageKN<BK, BN, WM * WN * 32>, Stage<BN, BK, WM * WN * 32>>;
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp / WN, wn = warp % WN;
	const int q = ( lane >> 2 ) & 3, qm = q & 1, qn = q >> 1, p = ( lane & 3 ) + 4 * ( lane >> 4 );
	const int arow = wm * TM + frag_line( qm, p ), wrow = wn * TN + frag_line( qn, p );
	const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
	// this split's k range: the operands start at its first value, k is its length
	const int kz0 = blockIdx.z * k_per_split;
	a += kz0;
	w += WKN ? static_cast<size_t>( kz0 ) * ldw : kz0;
	k = min( k_per_split, k - kz0 );
	SA sa;
	SW sw;
	sa.load( a, lda, m, k, m0, 0 );
	sw.load( w, ldw, n, k, n0, 0 );
	sa.store( &as[0][0][0] );
	sw.store( ws );
	__syncthreads();
	float acc[MB][NB][8] = {};
	constexpr int KK = kBK / 4;   // mma k steps per tile
	// a fragment row's swizzle is the same for all of a lane's fragments (they are 16 rows apart)
	const uint32_t afrag = static_cast<uint32_t>( __cvta_generic_to_shared( &as[0][arow][0] ) );
	// [k, n] weights: k row lane % 4 of the step, the quad pair's columns 4 qn + 8 (lane >= 16) + 0..3 of each 16-column fragment (chunk
	// wn TN / 8 + 2 j + (lane >= 16), swizzled by the row)
	const uint32_t wfrag = static_cast<uint32_t>( __cvta_generic_to_shared( WKN ? ws + ( lane & 3 ) * BN + 4 * qn : ws + wrow * BK ) );
	const int wchunk = wn * TN / 8 + ( lane >> 4 );
	const int akey = SA::swz( arow, 0 ), wkey = SA::swz( wrow, 0 );
	uint2 av[2][MB], bv[2][NB];
	auto lds = []( uint2& v, uint32_t addr ) { asm volatile( "ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"( v.x ), "=r"( v.y ) : "r"( addr ) ); };
	auto frags = [&]( int stage, int kk, int buf ) {
		const int ao = 2 * ( 8 * ( ( kk >> 1 ) ^ akey ) + 4 * ( kk & 1 ) ), wo = 2 * ( 8 * ( ( kk >> 1 ) ^ wkey ) + 4 * ( kk & 1 ) );
		#pragma unroll
		for ( int i = 0; i < MB; ++i ) lds( av[buf][i], afrag + 2 * ( stage * BM * BK + 16 * i * BK ) + ao );
		#pragma unroll
		for ( int j = 0; j < NB; ++j ) {
			if constexpr ( WKN ) lds( bv[buf][j], wfrag + 2 * ( stage * WSTAGE + 4 * kk * BN + 8 * ( ( wchunk + 2 * j ) ^ ( lane & 3 ) ) ) );
			else lds( bv[buf][j], wfrag + 2 * ( stage * WSTAGE + 16 * j * BK ) + wo );
		}
	};
	frags( 0, 0, 0 );
	const int steps = ( k + kBK - 1 ) / kBK;
	if ( steps > 1 ) {
		sa.load( a, lda, m, k, m0, kBK );
		sw.load( w, ldw, n, k, n0, kBK );
	}
	for ( int s = 0; s < steps; ++s ) {
		const int cur = s & 1;
		const bool more = s + 1 < steps;
		#pragma unroll
		for ( int kk = 0; kk < KK; ++kk ) {
			// the next step's tiles load during the first k step; the next fragments load while this one multiplies, and before the step's
			// last the next step's tiles go to the other buffer (its last reads were before the previous barrier) for its first fragments
			if ( kk + 1 < KK ) frags( cur, kk + 1, ( kk + 1 ) & 1 );
			else if ( more ) {
				sa.store( &as[cur ^ 1][0][0] );
				sw.store( ws + ( cur ^ 1 ) * WSTAGE );
				__syncthreads();
				frags( cur ^ 1, 0, 0 );
				if ( s + 2 < steps ) {
					sa.load( a, lda, m, k, m0, ( s + 2 ) * kBK );
					sw.load( w, ldw, n, k, n0, ( s + 2 ) * kBK );
				}
			}
			const int b = kk & 1;
			#pragma unroll
			for ( int i = 0; i < MB; ++i ) {
				#pragma unroll
				for ( int j = 0; j < NB; ++j ) {
					if constexpr ( WKN ) mma884_rr( acc[i][j], av[b][i].x, av[b][i].y, bv[b][j].x, bv[b][j].y );
					else mma884( acc[i][j], av[b][i].x, av[b][i].y, bv[b][j].x, bv[b][j].y );
				}
			}
		}
	}
	// Epilogue through shared memory, one 16-row fragment block of the warp's tile at a time: the accumulators go to a per-warp FP32 stage,
	// then each lane writes 8 consecutive columns of a row (16-byte stores where the row and alignment allow)
	__syncthreads();   // the last step's fragments are read: the tiles' memory is free
	constexpr int SLD = TN + 4;
	static_assert( WM * WN * 16 * SLD * 4 <= 2 * ( BM * BK + WSTAGE ) * 2, "epilogue stage fits in the tiles' memory" );
	float* stage = reinterpret_cast<float*>( smem ) + warp * 16 * SLD;
	const int r0 = ( lane & 1 ) + 4 * ( lane >> 4 ), c0 = lane & 2;
	constexpr int LPR = TN / 8, RPP = 32 / LPR;   // lanes per row, rows per pass
	const int er = lane / LPR, ec = 8 * ( lane % LPR );
	const bool split = gridDim.z > 1;
	#pragma unroll
	for ( int i = 0; i < MB; ++i ) {
		#pragma unroll
		for ( int j = 0; j < NB; ++j ) {
			#pragma unroll
			for ( int h = 0; h < 2; ++h ) {
				#pragma unroll
				for ( int rr = 0; rr < 2; ++rr ) {
					const int lr = frag_line( qm, r0 + 2 * rr ), lc = 16 * j + frag_line( qn, c0 + 4 * h );
					*reinterpret_cast<float2*>( stage + lr * SLD + lc ) = make_float2( acc[i][j][4 * h + 2 * rr], acc[i][j][4 * h + 2 * rr + 1] );
				}
			}
		}
		__syncwarp();
		#pragma unroll
		for ( int pr = 0; pr < 16; pr += RPP ) {
			const int lr = pr + er, row = m0 + wm * TM + 16 * i + lr, col = n0 + wn * TN + ec;
			if ( row >= m || col >= n ) continue;
			float v[8];
			*reinterpret_cast<float4*>( v ) = *reinterpret_cast<const float4*>( stage + lr * SLD + ec );
			*reinterpret_cast<float4*>( v + 4 ) = *reinterpret_cast<const float4*>( stage + lr * SLD + ec + 4 );
			const int cnt = min( 8, n - col );
			if ( split ) {
				float* pp = part + ( static_cast<size_t>( blockIdx.z ) * m + row ) * n + col;
				if ( cnt == 8 && ( reinterpret_cast<uintptr_t>( pp ) & 15 ) == 0 ) {
					reinterpret_cast<float4*>( pp )[0] = *reinterpret_cast<float4*>( v );
					reinterpret_cast<float4*>( pp )[1] = *reinterpret_cast<float4*>( v + 4 );
				}
				else for ( int e = 0; e < cnt; ++e ) pp[e] = v[e];
				continue;
			}
			const size_t o = static_cast<size_t>( row ) * ldc + col;
			if constexpr ( F32 ) {
				float* po = static_cast<float*>( out ) + o;
				if ( cnt == 8 && ( reinterpret_cast<uintptr_t>( po ) & 15 ) == 0 ) {
					float4* p4 = reinterpret_cast<float4*>( po );
					if ( beta != 0.0f ) {
						const float4 a0 = p4[0], a1 = p4[1];
						v[0] += beta * a0.x; v[1] += beta * a0.y; v[2] += beta * a0.z; v[3] += beta * a0.w;
						v[4] += beta * a1.x; v[5] += beta * a1.y; v[6] += beta * a1.z; v[7] += beta * a1.w;
					}
					p4[0] = *reinterpret_cast<float4*>( v );
					p4[1] = *reinterpret_cast<float4*>( v + 4 );
				}
				else for ( int e = 0; e < cnt; ++e ) po[e] = beta != 0.0f ? v[e] + beta * po[e] : v[e];
			}
			else {
				half* po = static_cast<half*>( out ) + o;
				if ( cnt == 8 && ( reinterpret_cast<uintptr_t>( po ) & 15 ) == 0 ) {
					uint4 pk;
					half2* h2 = reinterpret_cast<half2*>( &pk );
					if ( beta != 0.0f ) {
						const uint4 old = *reinterpret_cast<const uint4*>( po );
						const half2* o2 = reinterpret_cast<const half2*>( &old );
						#pragma unroll
						for ( int e = 0; e < 4; ++e ) { const float2 f = __half22float2( o2[e] ); v[2 * e] += beta * f.x; v[2 * e + 1] += beta * f.y; }
					}
					#pragma unroll
					for ( int e = 0; e < 4; ++e ) h2[e] = __floats2half2_rn( v[2 * e], v[2 * e + 1] );
					*reinterpret_cast<uint4*>( po ) = pk;
				}
				else for ( int e = 0; e < cnt; ++e ) po[e] = __float2half_rn( beta != 0.0f ? v[e] + beta * __half2float( po[e] ) : v[e] );
			}
		}
		__syncwarp();
	}

}

// out = beta * out + the sum of the splits' partials, 4 columns per thread
template <bool F32>
__global__ void split_reduce_kernel( const float* __restrict__ part, void* __restrict__ out, int ldc, int m, int n, int splits, float beta ) {
	const int quads = ( n + 3 ) / 4, i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= m * quads ) return;
	const int row = i / quads, col = 4 * ( i % quads ), cnt = min( 4, n - col );
	const size_t plane = static_cast<size_t>( m ) * n;
	const float* pp = part + static_cast<size_t>( row ) * n + col;
	float v[4] = {};
	if ( cnt == 4 && ( n & 3 ) == 0 ) {
		for ( int z = 0; z < splits; ++z ) {
			const float4 t = *reinterpret_cast<const float4*>( pp + z * plane );
			v[0] += t.x; v[1] += t.y; v[2] += t.z; v[3] += t.w;
		}
	}
	else {
		for ( int z = 0; z < splits; ++z ) {
			for ( int e = 0; e < cnt; ++e ) v[e] += pp[z * plane + e];
		}
	}
	const size_t o = static_cast<size_t>( row ) * ldc + col;
	if constexpr ( F32 ) {
		float* po = static_cast<float*>( out ) + o;
		for ( int e = 0; e < cnt; ++e ) po[e] = beta != 0.0f ? v[e] + beta * po[e] : v[e];
	}
	else {
		half* po = static_cast<half*>( out ) + o;
		if ( cnt == 4 && beta == 0.0f && ( reinterpret_cast<uintptr_t>( po ) & 7 ) == 0 ) {
			half2 h[2] = { __floats2half2_rn( v[0], v[1] ), __floats2half2_rn( v[2], v[3] ) };
			*reinterpret_cast<uint2*>( po ) = *reinterpret_cast<const uint2*>( h );
		}
		else for ( int e = 0; e < cnt; ++e ) po[e] = __float2half_rn( beta != 0.0f ? v[e] + beta * __half2float( po[e] ) : v[e] );
	}
}

// The split-k partials, per stream (GEMMs on one stream run in order; two streams may overlap), grown on demand. A buffer is never freed (a
// captured graph may still use it), and growing only allocates, which a graph capture allows: whether a GEMM splits depends on its shape
// alone, so a captured product sums in the same order as the eager one
constexpr size_t kSplitFloats = size_t( 16 ) << 20;
struct Space { float* part = nullptr; size_t cap = 0; };
std::mutex g_space_mu;
std::map<std::pair<int, cudaStream_t>, Space> g_spaces;
// Bumped by reset_caches: each launch() configuration then sets its shared-memory attribute and reads its occupancy again
std::atomic<int> g_launch_epoch{ 1 };

float* split_workspace( size_t floats, cudaStream_t stream ) {
	auto& mu = g_space_mu;
	auto& spaces = g_spaces;
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( mu );
	Space& sp = spaces[{ dev, stream }];
	if ( floats > sp.cap ) {
		const size_t cap = std::min( kSplitFloats, std::max( floats, 2 * sp.cap ) );
		CUDA_CHECK( cudaMalloc( &sp.part, cap * sizeof( float ) ) );
		sp.cap = cap;
	}
	return sp.part;
}

int sm_count() {
	static int sms[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( sms[dev] == 0 ) CUDA_CHECK( cudaDeviceGetAttribute( &sms[dev], cudaDevAttrMultiProcessorCount, dev ) );
	return sms[dev];
}

// The split-k model, measured on a V100 at the 200 W cap: the multiply rate of 80 SMs with two 128 x 128 blocks each (TFLOPS), the time of
// a partial wave's block alone on its SM and of two sharing one (in tile times), and the bandwidth the split partials move at (GB/s)
constexpr double kSmRate = 70.0, kLone = 1.17, kShared = 1.56, kPartBandwidth = 700.0;

template <int BM, int BN, int BK, int WM, int WN, bool WKN>
void launch( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	constexpr int threads = WM * WN * 32, smem = 2 * ( BM * BK + w_stage_halves<WKN>( BN, BK ) ) * static_cast<int>( sizeof( half ) );
	auto kernel = out_f32 ? gemm_tn_kernel<BM, BN, BK, WM, WN, true, WKN> : gemm_tn_kernel<BM, BN, BK, WM, WN, false, WKN>;
	static int per_sm[2][64] = {}, epoch[2][64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	if ( epoch[out_f32][dev] != g_launch_epoch.load() ) {
		epoch[out_f32][dev] = g_launch_epoch.load();
		CUDA_CHECK( cudaFuncSetAttribute( kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem ) );
		CUDA_CHECK( cudaOccupancyMaxActiveBlocksPerMultiprocessor( &per_sm[out_f32][dev], kernel, threads, smem ) );
	}
	const int tiles = ( ( n + BN - 1 ) / BN ) * ( ( m + BM - 1 ) / BM ), slots = per_sm[out_f32][dev] * sm_count();
	// split along k when it spreads the work over the SMs more evenly. Cost of a launch in tile times (one tile on an SM to itself, with the
	// whole GPU busy): full waves of block slots, then a partial wave whose blocks run alone on their SMs or share them; each block is
	// 1 / splits of a tile's work; plus the partials' round trip (written and read back in FP32); at least 256 values of k per split.
	// Small tiles stream the weights and count every slot as a full share
	const int sms = sm_count(), per = per_sm[out_f32][dev];
	const bool big = BM * BN >= 128 * 128;
	// rows rounded up to whole tiles: every row count with the same tiles splits the same way, so an output row does not depend on how many
	// rows ride along (a decode step and a verify of the same token agree)
	const size_t mt = static_cast<size_t>( ( m + BM - 1 ) / BM ) * BM;
	const double tile_s = 2.0 * BM * BN * k / ( kSmRate * 1e12 / 80 ), part_s = 2.0 * 4.0 * mt * n / ( kPartBandwidth * 1e9 );
	auto cost = [&]( int sp ) {
		const int units = tiles * sp, full = units / slots, rest = units % slots;
		const double tail = rest == 0 ? 0.0 : !big ? per : rest <= sms ? kLone : kShared;
		return ( full * per + tail ) * tile_s / sp + ( sp > 1 ? part_s * sp : 0.0 );
	};
	int splits = 1;
	double best = cost( 1 );
	for ( int sp = 2; sp <= std::min( 16, k / 256 ) && sp * mt * n <= kSplitFloats; ++sp ) {
		const double t = cost( sp );
		if ( t < best ) { best = t; splits = sp; }
	}
	int k_per_split = ( ( k + splits - 1 ) / splits + 7 ) / 8 * 8, used = ( k + k_per_split - 1 ) / k_per_split;
	float* part = used > 1 ? split_workspace( static_cast<size_t>( used ) * m * n, stream ) : nullptr;
	const dim3 grid( ( n + BN - 1 ) / BN, ( m + BM - 1 ) / BM, used );
	kernel<<<grid, threads, smem, stream>>>( a, lda, w, ldw, out, ldc, m, n, k, beta, part, k_per_split );
	if ( used > 1 ) {
		const int quads = m * ( ( n + 3 ) / 4 );
		if ( out_f32 ) split_reduce_kernel<true><<<( quads + 255 ) / 256, 256, 0, stream>>>( part, out, ldc, m, n, used, beta );
		else split_reduce_kernel<false><<<( quads + 255 ) / 256, 256, 0, stream>>>( part, out, ldc, m, n, used, beta );
	}
}

template <bool WKN>
void dispatch( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	// tile by row count, from a sweep over the language models' projection shapes (20 - 2048 rows)
	if ( m <= 32 ) launch<32, 64, 64, 1, 2, WKN>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
	else if ( m <= 64 || ( m > 128 && m <= 192 ) ) launch<64, 64, 64, 2, 2, WKN>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
	else if ( m <= 128 ) launch<64, 128, 64, 2, 2, WKN>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
	else launch<128, 128, 32, 2, 2, WKN>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

} // namespace

void Gemm::tn( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	ENGINE_ASSERT( k % 8 == 0 && lda % 8 == 0 && ldw % 8 == 0, "Gemm::tn: k and the operands' leading dimensions in 16-byte rows" );
	ENGINE_ASSERT( reinterpret_cast<uintptr_t>( a ) % 16 == 0 && reinterpret_cast<uintptr_t>( w ) % 16 == 0, "Gemm::tn: 16-byte aligned operands" );
	dispatch<false>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

void Gemm::nn( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	ENGINE_ASSERT( k % 8 == 0 && n % 8 == 0 && lda % 8 == 0 && ldw % 8 == 0, "Gemm::nn: k, n and the operands' leading dimensions in 16-byte rows" );
	ENGINE_ASSERT( reinterpret_cast<uintptr_t>( a ) % 16 == 0 && reinterpret_cast<uintptr_t>( w ) % 16 == 0, "Gemm::nn: 16-byte aligned operands" );
	dispatch<true>( a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

} // namespace engine::modules
namespace engine::modules {

void Gemm::reset_caches() {
	std::lock_guard<std::mutex> lk( g_space_mu );
	g_spaces.clear();   // a device reset freed them
	g_launch_epoch.fetch_add( 1 );
}

} // namespace engine::modules

#else

// ROCm: the kernel above is Volta PTX; hipBLAS does the products (one handle per device)
namespace engine::modules {

namespace {

cublasHandle_t blas() {
	static std::mutex mu;
	static cublasHandle_t h[64] = {};
	int dev = 0;
	cudaGetDevice( &dev );
	std::lock_guard<std::mutex> lk( mu );
	if ( !h[dev] ) ENGINE_ASSERT( cublasCreate( &h[dev] ) == CUBLAS_STATUS_SUCCESS, "Gemm: hipblasCreate failed" );
	return h[dev];
}

void gemm( bool wkn, const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	cublasHandle_t h = blas();
	cublasSetStream( h, stream );
	const float one = 1.0f;
	// row-major out[m, n] is column-major [n, m]: out^T = op(w) a^T
	const cublasStatus_t st = cublasGemmEx( h, wkn ? CUBLAS_OP_N : CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, w, CUDA_R_16F, ldw, a, CUDA_R_16F, lda, &beta, out,
		out_f32 ? CUDA_R_32F : CUDA_R_16F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
	ENGINE_ASSERT( st == CUBLAS_STATUS_SUCCESS, "Gemm: hipblasGemmEx failed" );
}

} // namespace

void Gemm::tn( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	gemm( false, a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

void Gemm::reset_caches() {}

void Gemm::nn( const half* a, int lda, const half* w, int ldw, void* out, int ldc, bool out_f32, int m, int n, int k, float beta, cudaStream_t stream ) {
	gemm( true, a, lda, w, ldw, out, ldc, out_f32, m, n, k, beta, stream );
}

} // namespace engine::modules

#endif
