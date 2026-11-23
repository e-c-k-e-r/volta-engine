// FP8 routed experts on CUDA (FP8MoE in engine/modules/fp8_moe.h): the E4M3 weights of the checkpoint with their 128 x 128 F32 block scales,
// dequantized inside the expert GEMM. A pass's routes are grouped by expert into work items of up to 16 or 32 routes (moe_items.cuh); per item
// a block takes a tile of 32 output rows (inside one 128-row scale block), and per 128-wide input step decodes the tile's E4M3 bytes to FP16
// in shared memory, copies the routes' input rows beside them and multiplies on the tensor cores (WMMA, FP32 accumulation); each step's sum
// is scaled by the tile's block scale of that step into the accumulator.
#include "engine/modules/fp8_moe.h"
#include "engine/backend/runtime.h"
#include "../moe_items.cuh"
#include <mma.h>
#include <algorithm>

namespace engine::modules {

namespace {

constexpr int kRowsPerPass = 1024;   // rows of a pass
constexpr int kRoutesMax = 32;       // routes of an item: 16, or 32 when a pass gives the local experts more than about 12 routes each
constexpr int kTileN = 32;           // output rows of a block (of gate and up together under GU)
constexpr int kThreads = 256;        // 8 warps: two row halves x four quarters of a step's inputs
constexpr int kStep = 128;           // inputs per step: one scale block
constexpr int kLd = kStep + 16;      // halves per shared-memory row (rows 32-byte aligned for WMMA)
constexpr int kMaxSteps = 64;        // the widest input: 8192

// Four E4M3 bytes as four FP16 halves (bytes 0, 1 in lo, 2, 3 in hi): the sign to bit 15, exponent and mantissa to bits 7..13, which is the
// value times 2^-8 exactly (subnormals included; the NaN codes, absent from weights, become finite)
__device__ __forceinline__ void e4m3x4_h( uint32_t w, uint32_t& lo, uint32_t& hi ) {
	const uint32_t a = __byte_perm( w, 0, 0x4140 ), b = __byte_perm( w, 0, 0x4342 );   // bytes 0, 1 (2, 3) in the low bytes of the two halves
	lo = ( ( a & 0x00800080u ) << 8 ) | ( ( a & 0x007f007fu ) << 7 );
	hi = ( ( b & 0x00800080u ) << 8 ) | ( ( b & 0x007f007fu ) << 7 );
}

__global__ void fp8_dispatch_kernel( const int* __restrict__ topk, int routes, int first, int num_local, Item* __restrict__ items, int* __restrict__ n_items,
	int* __restrict__ sorted, uint32_t* __restrict__ heat, int item_routes ) {
	moe_group_routes( topk, routes, first, num_local, items, n_items, sorted, heat, item_routes );
}

// out[route][row] = expert row `row` . the route's FP16 input row (the token's row of x, or the route's own row of the activation) for the item's
// routes and the tile's rows. GU: the tile's first 16 rows from the gate and the last 16 the same rows of the up projection, the epilogue writing
// SiLU( gate ) * up to out16 [route][n]; else the down projection's FP32 partials scaled by the route's weight (part32), or FP16 rows (out16)
template <bool GU, int RT>
__global__ __launch_bounds__( kThreads, 4 )
void fp8_tile_kernel( const uint8_t* const* __restrict__ codes, const float* const* __restrict__ scales, const uint8_t* const* __restrict__ codes_up,
	const float* const* __restrict__ scales_up, const Item* __restrict__ items, const int* __restrict__ n_items, const int* __restrict__ sorted,
	const half* __restrict__ in, bool per_token, int k, int n, const half* __restrict__ topw, int top_k, float* __restrict__ part32, half* __restrict__ out16,
	float act_limit ) {
	using namespace nvcuda;
	if ( static_cast<int>( blockIdx.y ) >= *n_items ) return;
	extern __shared__ __align__( 32 ) unsigned char fp8_smem[];
	half* sW = reinterpret_cast<half*>( fp8_smem );   // [kTileN][kLd]
	half* sX = sW + kTileN * kLd;                     // [RT][kLd]
	__shared__ int s_route[RT];
	__shared__ float s_scale[2][kMaxSteps];           // the tile's block scale per step (times 256: the decode's 2^-8), gate and up
	const Item it = items[blockIdx.y];
	constexpr int kRows = GU ? kTileN / 2 : kTileN;   // output rows of the block (of each of gate and up)
	const int row0 = blockIdx.x * kRows, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	const int steps = k / kStep;
	if ( threadIdx.x < RT ) s_route[threadIdx.x] = threadIdx.x < it.count ? sorted[it.begin + threadIdx.x] : -1;
	{
		const float* sg = scales[it.expert] + static_cast<size_t>( row0 / 128 ) * steps;
		const float* su = GU ? scales_up[it.expert] + static_cast<size_t>( row0 / 128 ) * steps : sg;
		for ( int i = threadIdx.x; i < 2 * steps; i += kThreads ) s_scale[i / steps][i % steps] = ( i < steps ? sg[i] : su[i - steps] ) * 256.0f;
	}
	// this thread's 16 weight bytes of each step: tile row tr (gate rows first under GU), bytes 16 c .. 16 c + 15 of the step; rows past n read zeros
	const int tr = threadIdx.x >> 3, c = threadIdx.x & 7;
	const int gr = row0 + ( GU ? tr % kRows : tr );
	const uint8_t* wsrc = gr < n ? ( GU && tr >= kRows ? codes_up : codes )[it.expert] + static_cast<size_t>( gr ) * k + 16 * c : nullptr;
	// the routes' input rows this thread copies (RT routes x 16 pieces of 8 halves a step)
	constexpr int XPER = RT * 16 / kThreads;
	const half* xsrc[XPER];
	#pragma unroll
	for ( int i = 0; i < XPER; ++i ) {
		const int v = threadIdx.x + i * kThreads, j = v >> 4, r = j < it.count ? sorted[it.begin + j] : -1;
		xsrc[i] = r >= 0 ? in + static_cast<size_t>( per_token ? r / top_k : r ) * k + ( v & 15 ) * 8 : nullptr;
	}
	uint4 wr = wsrc != nullptr ? *reinterpret_cast<const uint4*>( wsrc ) : make_uint4( 0, 0, 0, 0 );
	const int wn = ( warp & 1 ) * 16, kq = ( warp >> 1 ) * 32;   // the warp's rows and quarter of a step's inputs
	const int sidx = GU && wn >= kRows ? 1 : 0;
	constexpr int RF = RT / 16;   // 16-route fragments
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[RF];
	#pragma unroll
	for ( int f = 0; f < RF; ++f ) wmma::fill_fragment( acc[f], 0.0f );
	for ( int kb = 0; kb < steps; ++kb ) {
		{
			uint4 lo, hi;
			e4m3x4_h( wr.x, lo.x, lo.y );
			e4m3x4_h( wr.y, lo.z, lo.w );
			e4m3x4_h( wr.z, hi.x, hi.y );
			e4m3x4_h( wr.w, hi.z, hi.w );
			uint4* d = reinterpret_cast<uint4*>( sW + tr * kLd + 16 * c );
			d[0] = lo;
			d[1] = hi;
		}
		#pragma unroll
		for ( int i = 0; i < XPER; ++i ) {
			const int v = threadIdx.x + i * kThreads;
			const uint4 val = xsrc[i] != nullptr ? *reinterpret_cast<const uint4*>( xsrc[i] + kb * kStep ) : make_uint4( 0, 0, 0, 0 );
			*reinterpret_cast<uint4*>( sX + ( v >> 4 ) * kLd + ( v & 15 ) * 8 ) = val;
		}
		if ( kb + 1 < steps && wsrc != nullptr ) wr = *reinterpret_cast<const uint4*>( wsrc + ( kb + 1 ) * kStep );   // in flight during the multiply
		__syncthreads();
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> part[RF];
		#pragma unroll
		for ( int f = 0; f < RF; ++f ) wmma::fill_fragment( part[f], 0.0f );
		#pragma unroll
		for ( int kk = 0; kk < 32; kk += 16 ) {
			wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
			wmma::load_matrix_sync( b, sW + wn * kLd + kq + kk, kLd );
			#pragma unroll
			for ( int f = 0; f < RF; ++f ) {
				if ( f * 16 >= it.count ) break;
				wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
				wmma::load_matrix_sync( a, sX + f * 16 * kLd + kq + kk, kLd );
				wmma::mma_sync( part[f], a, b, part[f] );
			}
		}
		const float s = s_scale[sidx][kb];
		#pragma unroll
		for ( int f = 0; f < RF; ++f )
			#pragma unroll
			for ( int e = 0; e < part[f].num_elements; ++e ) acc[f].x[e] += s * part[f].x[e];
		__syncthreads();
	}
	// the four input quarters' sums through shared memory, added in order, to the routes' rows
	float* sC = reinterpret_cast<float*>( fp8_smem );   // [4][RT][kTileN + 4]
	constexpr int ldc = kTileN + 4;
	#pragma unroll
	for ( int f = 0; f < RF; ++f ) wmma::store_matrix_sync( sC + ( ( warp >> 1 ) * RT + f * 16 ) * ldc + wn, acc[f], ldc, wmma::mem_row_major );
	__syncthreads();
	for ( int v = threadIdx.x; v < RT * kRows; v += kThreads ) {
		const int j = v / kRows, cc = v % kRows, r = s_route[j], g = row0 + cc;
		if ( r < 0 || g >= n ) continue;
		const float* q = sC + j * ldc + cc;
		const float val = ( q[0] + q[RT * ldc] ) + ( q[2 * RT * ldc] + q[3 * RT * ldc] );
		if constexpr ( GU ) {
			const float* qu = q + kRows;
			float gv = __half2float( __float2half_rn( val ) );
			float uv = __half2float( __float2half_rn( ( qu[0] + qu[RT * ldc] ) + ( qu[2 * RT * ldc] + qu[3 * RT * ldc] ) ) );
			if ( act_limit > 0.0f ) {
				gv = fminf( gv, act_limit );
				uv = fmaxf( -act_limit, fminf( uv, act_limit ) );
			}
			out16[static_cast<size_t>( r ) * n + g] = __float2half_rn( gv / ( 1.0f + __expf( -gv ) ) * uv );
		}
		else if ( part32 != nullptr ) part32[static_cast<size_t>( r ) * n + g] = val * __half2float( topw[r] );
		else out16[static_cast<size_t>( r ) * n + g] = __float2half_rn( val );
	}
}

constexpr size_t tile_smem( int rt ) {
	return std::max( static_cast<size_t>( kTileN + rt ) * kLd * sizeof( half ), static_cast<size_t>( 4 ) * rt * ( kTileN + 4 ) * sizeof( float ) );
}

template <bool GU, int RT>
void launch_tile( const uint8_t* const* codes, const float* const* scales, const uint8_t* const* codes_up, const float* const* scales_up, const Item* items,
	const int* n_items, const int* sorted, int max_n, const half* in, bool per_token, int k, int n, const half* topw, int top_k, float* part32, half* out16,
	float act_limit, cudaStream_t stream ) {
	constexpr int rows = GU ? kTileN / 2 : kTileN;
	fp8_tile_kernel<GU, RT><<<dim3( ( n + rows - 1 ) / rows, max_n ), kThreads, tile_smem( RT ), stream>>>( codes, scales, codes_up, scales_up, items, n_items,
		sorted, in, per_token, k, n, topw, top_k, part32, out16, act_limit );
}

void tile_proj( int rt, const uint8_t* const* codes, const float* const* scales, const uint8_t* const* codes_up, const float* const* scales_up, const Item* items,
	const int* n_items, const int* sorted, int max_n, const half* in, bool per_token, int k, int n, const half* topw, int top_k, float* part32, half* out16,
	float act_limit, cudaStream_t stream ) {
	if ( codes_up != nullptr ) {
		if ( rt > 16 ) launch_tile<true, 32>( codes, scales, codes_up, scales_up, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream );
		else launch_tile<true, 16>( codes, scales, codes_up, scales_up, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream );
	}
	else {
		if ( rt > 16 ) launch_tile<false, 32>( codes, scales, nullptr, nullptr, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream );
		else launch_tile<false, 16>( codes, scales, nullptr, nullptr, items, n_items, sorted, max_n, in, per_token, k, n, topw, top_k, part32, out16, act_limit, stream );
	}
}

// The scratch of one pass: the activation in FP16, the down partials in FP32, the work items and the routes in expert order
struct Layout {
	size_t a, part, items, n_items, sorted, total;
};
Layout layout( int top_k, int hidden, int inter ) {
	const size_t routes = static_cast<size_t>( kRowsPerPass ) * top_k;
	auto up = []( size_t b ) { return ( b + 255 ) / 256 * 256; };
	Layout l{};
	l.a = 0;
	l.part = l.a + up( routes * inter * sizeof( half ) );
	l.items = l.part + up( routes * hidden * sizeof( float ) );
	l.n_items = l.items + up( routes * sizeof( Item ) );
	l.sorted = l.n_items + 256;
	l.total = l.sorted + up( routes * sizeof( int ) );
	return l;
}

} // namespace

size_t FP8MoE::scratch_bytes( int top_k, int hidden, int inter ) {
	return layout( top_k, hidden, inter ).total;
}

void FP8MoE::forward( const FP8ExpertTable& t, const half* x, int m, const int* topk_idx, const half* topk_w, int top_k, half* out, void* scratch, uint32_t* heat,
	float act_limit, cudaStream_t stream ) {
	ENGINE_ASSERT( t.hidden % kStep == 0 && t.inter % kStep == 0 && t.hidden <= kMaxSteps * kStep && t.inter <= kMaxSteps * kStep, "FP8MoE: widths must be whole 128-wide scale blocks, at most 8192" );
	char* ws = static_cast<char*>( scratch );
	const Layout l = layout( top_k, t.hidden, t.inter );
	auto* a16 = reinterpret_cast<half*>( ws + l.a );
	auto* part = reinterpret_cast<float*>( ws + l.part );
	auto* items = reinterpret_cast<Item*>( ws + l.items );
	auto* n_items = reinterpret_cast<int*>( ws + l.n_items );
	auto* sorted = reinterpret_cast<int*>( ws + l.sorted );
	for ( int r0 = 0; r0 < m; r0 += kRowsPerPass ) {
		const int rn = std::min( kRowsPerPass, m - r0 ), routes = rn * top_k;
		// 32-route items once the local experts average more than about 12 routes in the pass (fewer decodes, at a lower occupancy)
		const int global = t.num_global > 0 ? t.num_global : 4 * t.num_local;
		const int rt = routes > 12 * global ? kRoutesMax : 16, mi = std::min( routes, t.num_local + routes / rt );
		const int* ti = topk_idx + static_cast<size_t>( r0 ) * top_k;
		const half* tw = topk_w + static_cast<size_t>( r0 ) * top_k;
		fp8_dispatch_kernel<<<1, 256, dispatch_smem( t.num_local ), stream>>>( ti, routes, t.min_expert, t.num_local, items, n_items, sorted, heat, rt );
		tile_proj( rt, t.gate_c, t.gate_s, t.up_c, t.up_s, items, n_items, sorted, mi, x + static_cast<size_t>( r0 ) * t.hidden, true, t.hidden, t.inter, tw, top_k,
			nullptr, a16, act_limit, stream );
		tile_proj( rt, t.down_c, t.down_s, nullptr, nullptr, items, n_items, sorted, mi, a16, false, t.inter, t.hidden, tw, top_k, part, nullptr, 0.0f, stream );
		moe_combine_kernel<<<( rn * t.hidden + 255 ) / 256, 256, 0, stream>>>( part, ti, t.min_expert, t.num_local, out + static_cast<size_t>( r0 ) * t.hidden, rn, top_k,
			t.hidden );
	}
}

} // namespace engine::modules
