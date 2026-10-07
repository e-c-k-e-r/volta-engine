// Tier 2 parity for the DFlash2 backbone building blocks, against explicit FP32 CPU references.
//
// The dynamic grouped conv is the piece both reference implementations unit-test, and the one our old
// path faked, so it gets the closest reference here: a tap-by-tap loop over the block, not a
// reimplementation of the same indexing. Also checks the RMSNorm residual split and the plain-FP16
// GEMV that feeds the conv kernel projections.
#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/spec/dflash.h"
#include <cuda_fp16.h>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace engine::spec;

namespace {

unsigned g_seed = 12345u;
float rnd( float scale ) {
	g_seed = g_seed * 1664525u + 1013904223u;
	return ( static_cast<float>( g_seed & 0xffff ) / 65536.0f - 0.5f ) * 2.0f * scale;
}

// These kernels accumulate in FP32 and store FP16, so the honest bound is the distance to the
// correctly-rounded FP16 reference, measured in FP16 ULPs, not an absolute epsilon: at the real
// magnitudes (a 5120-long dot lands around +-10, where one FP16 quantum is 0.0098) an absolute gate
// measures the output format, not the arithmetic.
struct Diff { double max_abs = 0.0; double rms = 0.0; double max_ulp = 0.0; };

static double ulp_at( float v ) {
	const float a = fabsf( v );
	if ( a == 0.0f ) return 1e-7;
	// FP16 has a 10-bit mantissa; nextafter by one ulp of the output format.
	half h = __float2half_rn( a );
	const float up = __half2float( __hadd( h, __float2half( 1e-4f ) ) );
	const float dn = __half2float( __hadd( h, __float2half( -1e-4f ) ) );
	const double step = fmax( fmax( up - a, a - dn ), 1e-7 );
	return step;
}

Diff compare( const std::vector<float>& ref, const std::vector<half>& got ) {
	Diff d;
	double ss = 0.0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const float g = __half2float( got[i] );
		// Distance to the FP16 rounding of the reference, in ULPs of that value.
		const double rounded = __half2float( __float2half_rn( ref[i] ) );
		d.max_ulp = fmax( d.max_ulp, fabs( static_cast<double>( g ) - rounded ) / ulp_at( ref[i] ) );
		const double e = g - ref[i];
		d.max_abs = fmax( d.max_abs, fabs( e ) );
		ss += e * e;
	}
	d.rms = sqrt( ss / ( ref.size() ? ref.size() : 1 ) );
	return d;
}

// out[t, c] = sum_k (base[k, c] + dyn[t, k, c / group_size]) * x[t - k, c], k <= t.
// Reference is a literal tap-by-tap loop, matching test_grouped_conv_matches_reference upstream.
void test_dynconv( pod::MemoryArena& arena, int rows, int hidden, int taps, int group_size ) {
	const int groups = hidden / group_size;
	std::vector<half> h_x( static_cast<size_t>( rows ) * hidden );
	// The coefficients are FP32 in the engine; the block suite used to stage them FP16, which the
	// wrapper silently accepted and reinterpreted. Staging them FP32 makes this case match the engine.
	std::vector<float> h_dyn( static_cast<size_t>( rows ) * taps * groups );
	std::vector<half> h_base( static_cast<size_t>( taps ) * hidden );
	for ( auto& v : h_x ) v = __float2half_rn( rnd( 1.0f ) );
	for ( auto& v : h_dyn ) v = rnd( 0.25f );
	for ( auto& v : h_base ) v = __float2half_rn( rnd( 0.5f ) );

	std::vector<float> ref( static_cast<size_t>( rows ) * hidden, 0.0f );
	for ( int t = 0; t < rows; ++t ) {
		for ( int c = 0; c < hidden; ++c ) {
			const int g = c / group_size;
			// Mirror the kernel's arithmetic: w = base + dyn evaluated in float, accumulate in float.
			// Accumulating in double measures the reference's extra precision instead of the kernel.
			float acc = 0.0f;
			for ( int k = 0; k < taps; ++k ) {
				if ( k > t ) break;
				const float w = __half2float( h_base[static_cast<size_t>( k ) * hidden + c] )
					+ h_dyn[( ( static_cast<size_t>( t ) * taps ) + k ) * groups + g];
				acc += w * __half2float( h_x[( static_cast<size_t>( t ) - k ) * hidden + c] );
			}
			ref[static_cast<size_t>( t ) * hidden + c] = acc;
		}
	}

	engine::core::arena_reset( arena );
	int xs[2] = { rows, hidden };
	pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	int ds[3] = { rows, taps, groups };
	pod::TensorView dyn = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, ds, 3 );
	int bs[2] = { taps, hidden };
	pod::TensorView base = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, bs, 2 );
	pod::TensorView out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	CUDA_CHECK( cudaMemcpy( x.data, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( dyn.data, h_dyn.data(), h_dyn.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( base.data, h_base.data(), h_base.size() * 2, cudaMemcpyHostToDevice ) );

	dflash2_dynconv( x, dyn, base, out, group_size, false, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	std::vector<half> h_out( static_cast<size_t>( rows ) * hidden );
	CUDA_CHECK( cudaMemcpy( h_out.data(), out.data, h_out.size() * 2, cudaMemcpyDeviceToHost ) );

	const Diff d = compare( ref, h_out );
	// Report the worst element's own magnitude alongside the ULP count: ULP distance is meaningless
	// near zero, and this suite moved the coefficients to FP32 to match the engine, which shifts where
	// the worst case lands.
	size_t wi = 0;
	double worst_ulp = 0.0;
	for ( size_t i = 0; i < ref.size(); ++i ) {
		const float g = __half2float( h_out[i] );
		const double rounded = __half2float( __float2half_rn( ref[i] ) );
		const double u = fabs( static_cast<double>( g ) - rounded ) / ulp_at( ref[i] );
		if ( u > worst_ulp ) { worst_ulp = u; wi = i; }
	}
	std::printf( "  dynconv   rows=%d hidden=%d taps=%d gs=%-3d | max %.3e rms %.3e ulp %.2f"
		" (worst at ref=%.6g)\n",
		rows, hidden, taps, group_size, d.max_abs, d.rms, d.max_ulp, ref[wi] );
	std::fflush( stdout );
	ENGINE_ASSERT( d.max_ulp <= 4.0, "dynamic grouped conv deviates from the tap-by-tap reference" );
}

// sum = x + residual_in; normed = rmsnorm(sum) * (w + bias); residual_out = sum
// BF16 round-to-nearest-even, kept in FP32. Mirrors DFlashSM70 and the kernel's dflash_bf16_rne.
static float ref_bf16_rne( float v ) {
	uint32_t b;
	std::memcpy( &b, &v, sizeof( b ) );
	b = ( b + 0x7FFFu + ( ( b >> 16 ) & 1u ) ) & 0xFFFF0000u;
	float o;
	std::memcpy( &o, &b, sizeof( o ) );
	return o;
}

void test_rmsnorm_residual( pod::MemoryArena& arena, int rows, int hidden ) {
	std::vector<half> h_x( static_cast<size_t>( rows ) * hidden );
	std::vector<half> h_r( static_cast<size_t>( rows ) * hidden );
	std::vector<half> h_w( hidden );
	for ( auto& v : h_x ) v = __float2half_rn( rnd( 1.0f ) );
	for ( auto& v : h_r ) v = __float2half_rn( rnd( 1.0f ) );
	for ( auto& v : h_w ) v = __float2half_rn( 0.5f + rnd( 0.5f ) );

	std::vector<float> ref_normed( static_cast<size_t>( rows ) * hidden );
	std::vector<float> ref_res( static_cast<size_t>( rows ) * hidden );
	const float eps = 1e-6f, bias = 0.0f;
	for ( int t = 0; t < rows; ++t ) {
		double ss = 0.0;
		for ( int c = 0; c < hidden; ++c ) {
			// BF16 round-to-nearest-even on the transport, matching DFlashSM70RMSNorm and the kernel.
			const float s = ref_bf16_rne( __half2float( h_x[static_cast<size_t>( t ) * hidden + c] )
				+ __half2float( h_r[static_cast<size_t>( t ) * hidden + c] ) );
			ref_res[static_cast<size_t>( t ) * hidden + c] = s;
			ss += static_cast<double>( s ) * s;
		}
		const float inv = rsqrtf( static_cast<float>( ss / hidden ) + eps );
		for ( int c = 0; c < hidden; ++c ) {
			const float s = ref_res[static_cast<size_t>( t ) * hidden + c];
			const float wv = ref_bf16_rne( __half2float( h_w[c] ) + bias );
			ref_normed[static_cast<size_t>( t ) * hidden + c] = ref_bf16_rne( s * wv * inv );
		}
	}

	engine::core::arena_reset( arena );
	int xs[2] = { rows, hidden };
	pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	int ws[1] = { hidden };
	pod::TensorView w = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ws, 1 );
	pod::TensorView n = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	// The residual is FP32 in and out: it accumulates across all five layers and the reference keeps
	// range there deliberately, so the input is staged as FP32 too.
	int fs[2] = { rows, hidden };
	pod::TensorView ri = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, fs, 2 );
	pod::TensorView ro = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, fs, 2 );
	std::vector<float> h_ri_f( h_r.size() );
	for ( size_t i = 0; i < h_r.size(); ++i ) h_ri_f[i] = __half2float( h_r[i] );
	CUDA_CHECK( cudaMemcpy( x.data, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ri.data, h_ri_f.data(), h_ri_f.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( w.data, h_w.data(), h_w.size() * 2, cudaMemcpyHostToDevice ) );

	dflash_rmsnorm_residual( x, ri, w, n, ro, rows, eps, bias, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	std::vector<half> h_n( static_cast<size_t>( rows ) * hidden );
	std::vector<float> h_ro( static_cast<size_t>( rows ) * hidden );
	CUDA_CHECK( cudaMemcpy( h_n.data(), n.data, h_n.size() * 2, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( h_ro.data(), ro.data, h_ro.size() * 4, cudaMemcpyDeviceToHost ) );

	const Diff dn = compare( ref_normed, h_n );
	double max_res = 0.0;
	for ( size_t i = 0; i < ref_res.size(); ++i )
		max_res = fmax( max_res, fabs( static_cast<double>( h_ro[i] ) - ref_res[i] ) );
	std::printf( "  rmsnorm   rows=%d hidden=%-5d        | max %.3e rms %.3e ulp %.2f | residual max %.3e\n",
		rows, hidden, dn.max_abs, dn.rms, dn.max_ulp, max_res );
	std::fflush( stdout );
	ENGINE_ASSERT( dn.max_ulp <= 4.0, "RMSNorm residual split deviates from reference" );
	ENGINE_ASSERT( max_res < 1e-5, "residual stream deviates from x + residual_in" );
}

// out[m, n] = sum_k w[n, k] * x[m, k]
void test_f16_gemv_nt( pod::MemoryArena& arena, int m, int n, int k ) {
	std::vector<half> h_w( static_cast<size_t>( n ) * k );
	std::vector<half> h_x( static_cast<size_t>( m ) * k );
	for ( auto& v : h_w ) v = __float2half_rn( rnd( 0.25f ) );
	for ( auto& v : h_x ) v = __float2half_rn( rnd( 1.0f ) );

	std::vector<float> ref( static_cast<size_t>( m ) * n );
	for ( int i = 0; i < m; ++i ) {
		for ( int j = 0; j < n; ++j ) {
			double acc = 0.0;
			for ( int kk = 0; kk < k; ++kk )
				acc += static_cast<double>( __half2float( h_w[static_cast<size_t>( j ) * k + kk] ) )
					* __half2float( h_x[static_cast<size_t>( i ) * k + kk] );
			ref[static_cast<size_t>( i ) * n + j] = static_cast<float>( acc );
		}
	}

	engine::core::arena_reset( arena );
	int ws[2] = { n, k };
	pod::TensorView w = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ws, 2 );
	int xs[2] = { m, k };
	pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	int os[2] = { m, n };
	pod::TensorView out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, os, 2 );
	CUDA_CHECK( cudaMemcpy( w.data, h_w.data(), h_w.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( x.data, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice ) );

	dflash_f16_gemv_nt( w.ptr<const half>(), x.ptr<const half>(), out.ptr<half>(), m, n, k, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	std::vector<half> h_out( static_cast<size_t>( m ) * n );
	CUDA_CHECK( cudaMemcpy( h_out.data(), out.data, h_out.size() * 2, cudaMemcpyDeviceToHost ) );

	const Diff d = compare( ref, h_out );
	std::printf( "  f16 gemv  m=%d n=%d k=%-5d          | max %.3e rms %.3e ulp %.2f\n", m, n, k, d.max_abs, d.rms, d.max_ulp );
	std::fflush( stdout );
	ENGINE_ASSERT( d.max_ulp <= 4.0, "plain FP16 NT GEMV deviates from reference" );
}

// Reference: block-local per-head RMSNorm then NeoX RoPE over adjacent pairs, pass-through past the
// rotary width. theta^(-2p/rope_dim) with rope_dim = 2*pairs.
void test_qk_norm_rope( pod::MemoryArena& arena, int rows, int heads, int head_dim, int rope_dim ) {
	const int pairs = rope_dim / 2;
	const int pos0 = 137;
	const float theta = 1.0e7f, eps = 1e-6f;
	std::vector<half> h_x( static_cast<size_t>( rows ) * heads * head_dim );
	std::vector<half> h_w( head_dim );
	for ( auto& v : h_x ) v = __float2half_rn( rnd( 1.0f ) );
	for ( auto& v : h_w ) v = __float2half_rn( 0.5f + rnd( 0.5f ) );

	std::vector<float> cos_t( static_cast<size_t>( pos0 + rows ) * pairs );
	std::vector<float> sin_t( static_cast<size_t>( pos0 + rows ) * pairs );
	for ( int pos = 0; pos < pos0 + rows; ++pos ) {
		for ( int p = 0; p < pairs; ++p ) {
			const float inv = powf( theta, -2.0f * static_cast<float>( p ) / static_cast<float>( rope_dim ) );
			cos_t[static_cast<size_t>( pos ) * pairs + p] = cosf( pos * inv );
			sin_t[static_cast<size_t>( pos ) * pairs + p] = sinf( pos * inv );
		}
	}

	std::vector<float> ref( h_x.size() );
	for ( int r = 0; r < rows; ++r ) {
		for ( int h = 0; h < heads; ++h ) {
			const size_t base = ( static_cast<size_t>( r ) * heads + h ) * head_dim;
			double ss = 0.0;
			for ( int d = 0; d < head_dim; ++d ) {
				const double v = __half2float( h_x[base + d] );
				ss += v * v;
			}
			const float inv = rsqrtf( static_cast<float>( ss / head_dim ) + eps );
			std::vector<float> n( head_dim );
			for ( int d = 0; d < head_dim; ++d )
				n[d] = __half2float( h_x[base + d] ) * inv * ( __half2float( h_w[d] ) );
			const int pos = pos0 + r;
			for ( int d = 0; d < head_dim; ++d ) {
				if ( d < rope_dim ) {
					const int p = d >> 1;
					const float c = cos_t[static_cast<size_t>( pos ) * pairs + p];
					const float si = sin_t[static_cast<size_t>( pos ) * pairs + p];
					if ( ( d & 1 ) == 0 ) ref[base + d] = n[d] * c - n[d + 1] * si;
					else ref[base + d] = n[d] * c + n[d - 1] * si;
				}
				else ref[base + d] = n[d];
			}
		}
	}

	engine::core::arena_reset( arena );
	int xs[3] = { rows, heads, head_dim };
	pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 3 );
	int ws[1] = { head_dim };
	pod::TensorView w = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ws, 1 );
	int cs[2] = { pos0 + rows, pairs };
	pod::TensorView rc = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, cs, 2 );
	pod::TensorView rs = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, cs, 2 );
	CUDA_CHECK( cudaMemcpy( x.data, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( w.data, h_w.data(), h_w.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( rc.data, cos_t.data(), cos_t.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( rs.data, sin_t.data(), sin_t.size() * 4, cudaMemcpyHostToDevice ) );

	dflash_qk_norm_rope( x.ptr<half>(), w.ptr<const half>(), rc.ptr<const float>(), rs.ptr<const float>(),
		rows, heads, head_dim, pairs, pos0, eps, 0.0f, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	std::vector<half> h_out( h_x.size() );
	CUDA_CHECK( cudaMemcpy( h_out.data(), x.data, h_out.size() * 2, cudaMemcpyDeviceToHost ) );

	const Diff d = compare( ref, h_out );
	std::printf( "  qk+rope   rows=%d heads=%-3d hd=%d rope=%-3d  | max %.3e rms %.3e ulp %.2f\n",
		rows, heads, head_dim, rope_dim, d.max_abs, d.rms, d.max_ulp );
	std::fflush( stdout );
	ENGINE_ASSERT( d.max_ulp <= 4.0, "Q/K norm + RoPE deviates from reference" );
}

// Reference: plain softmax attention, non-causal over the block, GQA head mapping head / (q/kv).
void test_attn( pod::MemoryArena& arena, int rows, int q_heads, int kv_heads, int head_dim, int nctx ) {
	const int slots = nctx + rows;
	// The block contributes rows 1..rows-1. Row 0 is the anchor, whose position is already covered by
	// the context, so the block's key range starts one slot past own_slot.
	const int nblk = rows - 1;
	const int nkeys = nctx + nblk;
	const int own_slot = nctx;
	// Key kk -> cache slot: context keys keep their own index, block key j lands at own_slot + 1 + j.
	auto slot_of = [&] ( int kk ) { return kk < nctx ? kk : own_slot + 1 + ( kk - nctx ); };
	const int ss_slot = kv_heads * head_dim, ss_layer = kv_heads * head_dim, ss_head = head_dim;
	const float scale = rsqrtf( static_cast<float>( head_dim ) );

	std::vector<half> h_q( static_cast<size_t>( rows ) * q_heads * head_dim );
	std::vector<half> h_k( static_cast<size_t>( slots ) * q_heads * head_dim );
	std::vector<half> h_v( static_cast<size_t>( slots ) * q_heads * head_dim );
	for ( auto& v : h_q ) v = __float2half_rn( rnd( 1.0f ) );
	for ( auto& v : h_k ) v = __float2half_rn( rnd( 1.0f ) );
	for ( auto& v : h_v ) v = __float2half_rn( rnd( 1.0f ) );

	std::vector<float> ref( h_q.size() );
	for ( int r = 0; r < rows; ++r ) {
		for ( int h = 0; h < q_heads; ++h ) {
			const int g = h / ( q_heads / kv_heads );
			std::vector<float> scores( nkeys );
			float mx = -INFINITY;
			for ( int kk = 0; kk < nkeys; ++kk ) {
				const size_t kb = static_cast<size_t>( slot_of( kk ) ) * q_heads * head_dim + static_cast<size_t>( g ) * head_dim;
				double dot = 0.0;
				for ( int d = 0; d < head_dim; ++d )
					dot += static_cast<double>( __half2float( h_q[( ( static_cast<size_t>( r ) * q_heads + h ) * head_dim ) + d] ) )
						* __half2float( h_k[kb + d] );
				scores[kk] = static_cast<float>( dot ) * scale;
				mx = fmaxf( mx, scores[kk] );
			}
			double ls = 0.0;
			for ( int kk = 0; kk < nkeys; ++kk ) { scores[kk] = expf( scores[kk] - mx ); ls += scores[kk]; }
			for ( int d = 0; d < head_dim; ++d ) {
				double acc = 0.0;
				for ( int kk = 0; kk < nkeys; ++kk ) {
					const size_t vb = static_cast<size_t>( slot_of( kk ) ) * q_heads * head_dim + static_cast<size_t>( g ) * head_dim;
					acc += static_cast<double>( scores[kk] ) * __half2float( h_v[vb + d] );
				}
				ref[( static_cast<size_t>( r ) * q_heads + h ) * head_dim + d] = static_cast<float>( acc / ls );
			}
		}
	}

	engine::core::arena_reset( arena );
	int qs[3] = { rows, q_heads, head_dim };
	pod::TensorView q = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qs, 3 );
	int ks[4] = { slots, 1, kv_heads, head_dim };
	pod::TensorView kc = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ks, 4 );
	pod::TensorView vc = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ks, 4 );
	pod::TensorView out = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, qs, 3 );
	// Pack the reference K/V with the same stride the kernel is called with.
	std::vector<half> pk( static_cast<size_t>( slots ) * q_heads * head_dim, __float2half( 0.0f ) );
	std::vector<half> pv( pk.size(), __float2half( 0.0f ) );
	for ( int kk = 0; kk < slots; ++kk )
		for ( int g = 0; g < kv_heads; ++g )
			for ( int d = 0; d < head_dim; ++d ) {
				pk[static_cast<size_t>( kk ) * ss_slot + static_cast<size_t>( g ) * ss_head + d] =
					h_k[static_cast<size_t>( kk ) * q_heads * head_dim + static_cast<size_t>( g ) * head_dim + d];
				pv[static_cast<size_t>( kk ) * ss_slot + static_cast<size_t>( g ) * ss_head + d] =
					h_v[static_cast<size_t>( kk ) * q_heads * head_dim + static_cast<size_t>( g ) * head_dim + d];
			}
	CUDA_CHECK( cudaMemcpy( q.data, h_q.data(), h_q.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( kc.data, pk.data(), pk.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( vc.data, pv.data(), pv.size() * 2, cudaMemcpyHostToDevice ) );

	dflash_attn( q.ptr<const half>(), kc.ptr<const half>(), vc.ptr<const half>(), out.ptr<half>(),
		rows, q_heads, kv_heads, head_dim, nctx, 0, slots, 0, ss_slot, ss_layer, ss_head, own_slot, 1, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	std::vector<half> h_out( h_q.size() );
	CUDA_CHECK( cudaMemcpy( h_out.data(), out.data, h_out.size() * 2, cudaMemcpyDeviceToHost ) );

	const Diff d = compare( ref, h_out );
	std::printf( "  attn      rows=%d qh=%d kvh=%d hd=%d nctx=%-4d | max %.3e rms %.3e ulp %.2f\n",
		rows, q_heads, kv_heads, head_dim, nctx, d.max_abs, d.rms, d.max_ulp );
	std::fflush( stdout );
	ENGINE_ASSERT( d.max_ulp <= 8.0, "drafter attention deviates from softmax reference" );
}

// Per-row top-k over a padded vocabulary shard: scans [0, columns) of a row whose stride is
// row_stride, and offsets the indices by index_offset. Compared against a full sort of the same row.
void test_topk_shard( pod::MemoryArena& arena, int rows, int row_stride, int columns, int k, int index_offset ) {
	const int vocab = row_stride;
	std::vector<half> h_logits( static_cast<size_t>( rows ) * row_stride, __float2half( -100.0f ) );
	for ( int r = 0; r < rows; ++r ) {
		for ( int j = 0; j < columns; ++j ) {
			// Distinct values: a hashed column index plus sub-quantum jitter. Random logits over
			// 62080 columns would collide thousands of times and the comparison would then be
			// measuring tie-breaking rather than selection.
			const uint32_t hsh = ( static_cast<uint32_t>( j ) * 2654435761u ) ^ ( static_cast<uint32_t>( r ) * 40503u );
			const float v = static_cast<float>( hsh % 1000003u ) * 1e-3f + ( hsh & 0x3ffu ) * 1e-7f;
			h_logits[static_cast<size_t>( r ) * row_stride + j] = __float2half_rn( v );
		}
		// The pad region past `columns` holds -100, which would win any real logit if it were scanned.
	}

	engine::core::arena_reset( arena );
	int xs[2] = { rows, row_stride };
	pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	int vs[2] = { rows, k };
	pod::TensorView vals = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, vs, 2 );
	pod::TensorView idx = engine::core::arena_alloc_tensor( arena, pod::DType::INT32, vs, 2 );
	const int splits = ( columns + ( 256 * 16 ) - 1 ) / ( 256 * 16 );
	int ps[2] = { rows * splits * k, rows * splits * k };
	pod::TensorView pv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, ps, 2 );
	pod::TensorView pix = engine::core::arena_alloc_tensor( arena, pod::DType::INT32, ps, 2 );
	int cs[1] = { rows };
	pod::TensorView cnt = engine::core::arena_alloc_tensor( arena, pod::DType::INT32, cs, 1 );
	CUDA_CHECK( cudaMemcpy( x.data, h_logits.data(), h_logits.size() * 2, cudaMemcpyHostToDevice ) );

	dflash2_topk_shard( x.ptr<const half>(), rows, row_stride, columns, k,
		vals.ptr<float>(), idx.ptr<int>(), index_offset, pv.ptr<float>(), pix.ptr<int>(), cnt.ptr<int>(), nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );
	std::vector<float> h_v( static_cast<size_t>( rows ) * k );
	std::vector<int> h_i( static_cast<size_t>( rows ) * k );
	CUDA_CHECK( cudaMemcpy( h_v.data(), vals.data, h_v.size() * 4, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( h_i.data(), idx.data, h_i.size() * 4, cudaMemcpyDeviceToHost ) );

	// Reference: the k largest values among the first `columns`, sorted descending.
	//
	// The comparison is deliberately tie-break independent. FP16 carries an 11-bit mantissa, so a
	// vocabulary of 62080 columns collapses into a few thousand distinct logit values and the top
	// slots collide on exact ties. Any k-subset of a tied group is equally correct, so asserting a
	// specific winner would be testing the tie-break rule rather than the selection. What must hold:
	//   - the value multiset matches the reference exactly, and
	//   - every returned index actually carries the value returned beside it, so a value cannot be
	//     paired with a neighbour's index.
	double worst = 0.0;
	for ( int r = 0; r < rows; ++r ) {
		std::vector<float> all;
		all.reserve( columns );
		for ( int j = 0; j < columns; ++j )
			all.push_back( __half2float( h_logits[static_cast<size_t>( r ) * row_stride + j] ) );
		std::sort( all.begin(), all.end(), std::greater<float>() );
		for ( int t = 0; t < k; ++t ) {
			const float got_v = h_v[static_cast<size_t>( r ) * k + t];
			const int got_i = h_i[static_cast<size_t>( r ) * k + t];
			worst = fmax( worst, fabs( got_v - all[t] ) );
			const int col = got_i - index_offset;
			if ( col < 0 || col >= columns ) {
				worst = fmax( worst, 1e3 );   // an out-of-shard index
				continue;
			}
			const float at_ix = __half2float( h_logits[static_cast<size_t>( r ) * row_stride + col] );
			if ( at_ix != got_v ) worst = fmax( worst, 1e3 );   // value paired with the wrong index
		}
		// The selected set must be exactly the top-k: nothing below the k-th value may appear.
		const float kth = all[k - 1];
		for ( int t = 0; t < k; ++t ) {
			if ( h_v[static_cast<size_t>( r ) * k + t] < kth )
				worst = fmax( worst, 1e3 );
		}
	}
	std::printf( "  topk shard rows=%d stride=%-6d cols=%-6d k=%d off=%-7d | worst %.3e\n",
		rows, row_stride, columns, k, index_offset, worst );
	std::fflush( stdout );
	ENGINE_ASSERT( worst == 0.0, "shard top-k disagrees with the full-scan reference" );
}

// Distributed top-k: rank d scans its own shard, the merge must reproduce a top-k over the whole
// vocabulary. Shards get contiguous, disjoint column ranges with a non-zero index_offset, which is
// exactly how the vocabulary-parallel LM head slices the logits.
void test_topk_merge( pod::MemoryArena& arena, int tp, int rows, int k, int shard_real, int shard_cols ) {
	std::vector<std::vector<half>> h_shard( tp );
	std::vector<std::vector<float>> ref( rows );
	for ( int r = 0; r < rows; ++r ) ref[r].reserve( tp * shard_real );
	for ( int d = 0; d < tp; ++d ) {
		h_shard[d].assign( static_cast<size_t>( rows ) * shard_cols, __float2half( -100.0f ) );
		for ( int r = 0; r < rows; ++r ) {
			for ( int j = 0; j < shard_real; ++j ) {
				const uint32_t hsh = ( static_cast<uint32_t>( d * shard_real + j ) * 2654435761u )
					^ ( static_cast<uint32_t>( r ) * 40503u );
				const float v = static_cast<float>( hsh % 1000003u ) * 1e-3f + ( hsh & 0x3ffu ) * 1e-7f;
				const half hv = __float2half_rn( v );
				h_shard[d][static_cast<size_t>( r ) * shard_cols + j] = hv;
				// Reference must see the FP16-rounded value: that is what the device actually scans.
				ref[r].push_back( __half2float( hv ) );
			}
		}
	}

	engine::core::arena_reset( arena );
	int xs[2] = { rows, shard_cols };
	pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	int cs[3] = { tp, rows, 16 };
	pod::TensorView pv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, cs, 3 );
	pod::TensorView pix = engine::core::arena_alloc_tensor( arena, pod::DType::INT32, cs, 3 );
	int gs[2] = { rows, k };
	pod::TensorView gv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, gs, 2 );
	pod::TensorView gix = engine::core::arena_alloc_tensor( arena, pod::DType::INT32, gs, 2 );
	const int splits = ( shard_real + ( 256 * 16 ) - 1 ) / ( 256 * 16 );
	// dflash2_topk_shard instantiates the kernel at K = 16 regardless of k, and the kernel indexes its
	// partials as [row_id * splits + split] * 16. Sizing the scratch by k under-allocates whenever
	// k < 16 and the merge then reads past it.
	int ps[2] = { rows * splits * 16, rows * splits * 16 };
	pod::TensorView spv = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, ps, 2 );
	pod::TensorView spix = engine::core::arena_alloc_tensor( arena, pod::DType::INT32, ps, 2 );
	int cn[1] = { rows };
	pod::TensorView cnt = engine::core::arena_alloc_tensor( arena, pod::DType::INT32, cn, 1 );

	// Rank d's local top-k over its own columns, offset into the global token id space.
	for ( int d = 0; d < tp; ++d ) {
		CUDA_CHECK( cudaMemcpy( x.data, h_shard[d].data(), h_shard[d].size() * 2, cudaMemcpyHostToDevice ) );
		// Each rank writes its 16 partials at stride rows*16, regardless of k.
		dflash2_topk_shard( x.ptr<const half>(), rows, shard_cols, shard_real, k,
			pv.ptr<float>() + static_cast<size_t>( d ) * rows * 16,
			pix.ptr<int>() + static_cast<size_t>( d ) * rows * 16,
			d * shard_real, spv.ptr<float>(), spix.ptr<int>(), cnt.ptr<int>(), nullptr );
	}
	CUDA_CHECK( cudaDeviceSynchronize() );
	dflash2_topk_merge( pv.ptr<const float>(), pix.ptr<const int>(), tp, rows, k,
		gv.ptr<float>(), gix.ptr<int>(), 16, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );

	std::vector<float> h_gv( static_cast<size_t>( rows ) * k );
	std::vector<int> h_gi( static_cast<size_t>( rows ) * k );
	CUDA_CHECK( cudaMemcpy( h_gv.data(), gv.data, h_gv.size() * 4, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( h_gi.data(), gix.data, h_gi.size() * 4, cudaMemcpyDeviceToHost ) );

	double worst = 0.0;
	for ( int r = 0; r < rows; ++r ) {
		std::sort( ref[r].begin(), ref[r].end(), std::greater<float>() );
		// Compare as multisets: warp_topk breaks value ties by lane, not by global token id, and at
		// FP16 precision ties are pervasive. Sort what came back and compare elementwise.
		std::vector<float> got( h_gv.begin() + static_cast<size_t>( r ) * k,
			h_gv.begin() + static_cast<size_t>( r ) * k + k );
		std::sort( got.begin(), got.end(), std::greater<float>() );
		for ( int t = 0; t < k; ++t )
			worst = fmax( worst, fabs( got[t] - ref[r][t] ) );
		// Every returned index must address a shard that actually holds the value returned with it,
		// which catches a value paired with a neighbouring candidate's index.
		for ( int t = 0; t < k; ++t ) {
			const int got_i = h_gi[static_cast<size_t>( r ) * k + t];
			const int shard = got_i / shard_real;
			const int col = got_i % shard_real;
			if ( shard < 0 || shard >= tp || col < 0 || col >= shard_real ) {
				worst = fmax( worst, 1e3 );
				if ( r == 0 )
					std::printf( "    bounds slot%d: got_i %d -> shard %d col %d\n", t, got_i, shard, col );
				continue;
			}
			const float at_ix = __half2float( h_shard[shard][static_cast<size_t>( r ) * shard_cols + col] );
			if ( at_ix != h_gv[static_cast<size_t>( r ) * k + t] ) {
				worst = fmax( worst, 1e3 );
				std::printf( "    ix mismatch r%d slot%d: ix %d shard %d col %d at %.4f vs v %.4f\n",
					r, t, got_i, shard, col, at_ix, h_gv[static_cast<size_t>( r ) * k + t] );
			}
		}
	}
	std::printf( "  topk merge tp=%d rows=%d k=%-3d shard=%-6d | worst %.3e\n", tp, rows, k, shard_real, worst );
	std::fflush( stdout );
	ENGINE_ASSERT( worst == 0.0, "distributed top-k disagrees with the full-vocabulary reference" );
}

// The sandwich's kernel_projection emits both sides of a conv as one [rows, 2*taps*groups] FP16
// output, side 0 first. The split de-interleaves it into the [rows, taps, groups] FP32 tensor the
// grouped conv reads, groups innermost (per the reference's reshape to (rows, 2, taps, groups) and
// delta.unsqueeze(-1) against base.view(1, taps, groups, group_size)).
//
// This was unverified for the whole life of the feature. The decomposition was taps-innermost, so
// every conv coefficient in the engine was being read with tap and group transposed -- and because
// the end-to-end gate's ULP metric was also broken at the time, nothing reported it.
void test_conv_side_split( pod::MemoryArena& arena, int rows, int taps, int groups ) {
	const int blk = taps * groups;
	std::vector<half> h_proj( static_cast<size_t>( rows ) * 2 * blk );
	for ( auto& v : h_proj ) v = __float2half_rn( rnd( 1.0f ) );

	int ps[2] = { rows, 2 * blk };
	int ss[3] = { rows, taps, groups };

	double worst = 0.0;
	for ( int side_idx = 0; side_idx < 2; ++side_idx ) {
		engine::core::arena_reset( arena );
		pod::TensorView proj = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ps, 2 );
		pod::TensorView side = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, ss, 3 );
		CUDA_CHECK( cudaMemcpy( proj.data, h_proj.data(), h_proj.size() * 2, cudaMemcpyHostToDevice ) );

		dflash_conv_side_split( side.ptr<float>(), proj.ptr<const half>(), rows, taps, groups, side_idx, nullptr );
		CUDA_CHECK( cudaDeviceSynchronize() );
		std::vector<float> h_side( static_cast<size_t>( rows ) * blk );
		CUDA_CHECK( cudaMemcpy( h_side.data(), side.data, h_side.size() * 4, cudaMemcpyDeviceToHost ) );

		for ( int r = 0; r < rows; ++r ) {
			for ( int t = 0; t < taps; ++t ) {
				for ( int g = 0; g < groups; ++g ) {
					const float want = __half2float( h_proj[static_cast<size_t>( r ) * 2 * blk + side_idx * blk + t * groups + g] );
					const float got = h_side[static_cast<size_t>( r ) * blk + t * groups + g];
					worst = fmax( worst, fabs( static_cast<double>( got - want ) ) );
				}
			}
		}
	}

	std::printf( "  side split rows=%d taps=%d groups=%-4d blk=%-4d | worst %.3e\n",
		rows, taps, groups, blk, worst );
	std::fflush( stdout );
	ENGINE_ASSERT( worst == 0.0, "conv side split disagrees with the de-interleave reference" );
}

// dflash_rmsnorm_residual with residual_in and residual_out aliasing the same buffer.
//
// forward_backbone passes bb_residual_ as both at step (e) (and bb_normed_ as both x and normed_out at
// step (a)). Neither case was covered: the existing case stages separate FP32 buffers, so an
// in-place read-then-write hazard would pass it. This exercises the aliasing path against a reference
// computed from the pre-call values.
void test_rmsnorm_residual_alias( pod::MemoryArena& arena, int rows, int hidden, bool also_alias_x ) {
	std::vector<half> h_x( static_cast<size_t>( rows ) * hidden );
	std::vector<float> h_r( static_cast<size_t>( rows ) * hidden );
	std::vector<half> h_w( hidden );
	for ( auto& v : h_x ) v = __float2half_rn( rnd( 1.0f ) );
	for ( auto& v : h_r ) v = rnd( 1.0f );
	for ( auto& v : h_w ) v = __float2half_rn( 0.5f + rnd( 0.5f ) );
	const float eps = 1e-6f, bias = 0.0f;

	// Reference from the original values.
	std::vector<float> ref_normed( static_cast<size_t>( rows ) * hidden );
	std::vector<float> ref_res( static_cast<size_t>( rows ) * hidden );
	for ( int t = 0; t < rows; ++t ) {
		double ss = 0.0;
		for ( int c = 0; c < hidden; ++c ) {
			const float sv = ref_bf16_rne( __half2float( h_x[static_cast<size_t>( t ) * hidden + c] )
				+ h_r[static_cast<size_t>( t ) * hidden + c] );
			ref_res[static_cast<size_t>( t ) * hidden + c] = sv;
			ss += static_cast<double>( sv ) * sv;
		}
		const float inv = rsqrtf( static_cast<float>( ss / hidden ) + eps );
		for ( int c = 0; c < hidden; ++c ) {
			const float sv = ref_res[static_cast<size_t>( t ) * hidden + c];
			const float wv = ref_bf16_rne( __half2float( h_w[c] ) + bias );
			ref_normed[static_cast<size_t>( t ) * hidden + c] = ref_bf16_rne( sv * wv * inv );
		}
	}

	engine::core::arena_reset( arena );
	int xs[2] = { rows, hidden };
	pod::TensorView x = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	pod::TensorView n = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, xs, 2 );
	int ws[1] = { hidden };
	pod::TensorView w = engine::core::arena_alloc_tensor( arena, pod::DType::FP16, ws, 1 );
	int fs[2] = { rows, hidden };
	pod::TensorView ro = engine::core::arena_alloc_tensor( arena, pod::DType::FP32, fs, 2 );

	CUDA_CHECK( cudaMemcpy( x.data, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( ro.data, h_r.data(), h_r.size() * 4, cudaMemcpyHostToDevice ) );
	CUDA_CHECK( cudaMemcpy( w.data, h_w.data(), h_w.size() * 2, cudaMemcpyHostToDevice ) );

	// residual_in aliases residual_out; optionally x aliases normed_out as well, which is what
	// forward_backbone does at step (a).
	pod::TensorView rin = ro;
	pod::TensorView xv = x;
	pod::TensorView nv = n;
	if ( also_alias_x ) nv = x;
	dflash_rmsnorm_residual( xv, rin, w, nv, ro, rows, eps, bias, nullptr );
	CUDA_CHECK( cudaDeviceSynchronize() );

	std::vector<half> h_n( static_cast<size_t>( rows ) * hidden );
	std::vector<float> h_ro( static_cast<size_t>( rows ) * hidden );
	CUDA_CHECK( cudaMemcpy( h_n.data(), nv.data, h_n.size() * 2, cudaMemcpyDeviceToHost ) );
	CUDA_CHECK( cudaMemcpy( h_ro.data(), ro.data, h_ro.size() * 4, cudaMemcpyDeviceToHost ) );

	const Diff dn = compare( ref_normed, h_n );
	double max_res = 0.0;
	for ( size_t i = 0; i < ref_res.size(); ++i )
		max_res = fmax( max_res, fabs( static_cast<double>( h_ro[i] ) - ref_res[i] ) );
	std::printf( "  rmsnorm alias r=%d x=%s       | max %.3e rms %.3e ulp %.2f | residual max %.3e\n",
		rows, also_alias_x ? "yes" : "no", dn.max_abs, dn.rms, dn.max_ulp, max_res );
	std::fflush( stdout );
	ENGINE_ASSERT( dn.max_ulp <= 4.0, "aliased RMSNorm residual deviates from the reference" );
	ENGINE_ASSERT( max_res < 1e-5, "aliased residual stream deviates from x + residual_in" );
}

} // namespace

int main() {
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Tier 2: DFlash2 backbone blocks      \n" );
	std::printf( "========================================================\n" );

	setvbuf( stdout, nullptr, _IOLBF, 0 );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	pod::MemoryArena arena;
	std::fflush( stdout );
	ENGINE_ASSERT( engine::core::arena_create( arena, 0, 2ULL << 30 ), "arena" );

	// The block shape, plus a couple of geometry variants to catch indexing that only happens to work
	// for one size.
	test_dynconv( arena, 8, 5120, 2, 16 );
	test_dynconv( arena, 8, 5120, 2, 64 );
	test_dynconv( arena, 5, 512, 3, 16 );
	test_rmsnorm_residual( arena, 8, 5120 );
	test_rmsnorm_residual( arena, 3, 512 );
	// The real conv kernel projection geometry and the selector gate projection.
	test_f16_gemv_nt( arena, 8, 1280, 5120 );
	// Small k: the reduction is only two strided loads per lane here, and k == 64 was the
	// geometry the end-to-end gate diverged at.
	test_f16_gemv_nt( arena, 8, 128, 64 );
	test_f16_gemv_nt( arena, 8, 32, 64 );
	test_f16_gemv_nt( arena, 8, 128, 64 );
	// Rows beyond the query block: the context projection stages up to ctx_chunk_ = 256 rows,
	// which the previous one-warp-per-column kernel rejected outright at m > 8.
	test_f16_gemv_nt( arena, 9, 32, 64 );
	test_f16_gemv_nt( arena, 33, 16, 64 );
	test_f16_gemv_nt( arena, 256, 8, 64 );
	test_f16_gemv_nt( arena, 7, 256, 5120 );

	// Q/K norm + RoPE at the real geometry (rope_dim 64 of head_dim 128) and a full-rotary variant.
	test_qk_norm_rope( arena, 8, 32, 128, 64 );
	test_qk_norm_rope( arena, 8, 8, 128, 128 );
	test_qk_norm_rope( arena, 1, 4, 64, 32 );
	// Attention at the real GQA geometry plus small-window variants that change the key/warp mapping.
	test_attn( arena, 8, 32, 8, 128, 96 );
	test_attn( arena, 8, 32, 8, 128, 2040 );
	test_attn( arena, 1, 32, 8, 128, 3 );
	test_attn( arena, 8, 4, 4, 64, 17 );

	// Shard top-k, including the padded case the vocabulary-parallel LM head actually runs:
	// 62080 real columns in a 62208-wide shard, offset by the rank's slice of the vocabulary.
	test_topk_shard( arena, 7, 62208, 62080, 16, 0 );
	test_topk_shard( arena, 7, 62208, 62080, 16, 186240 );
	test_topk_shard( arena, 8, 256, 256, 16, 1000 );
	test_topk_shard( arena, 1, 1024, 1024, 4, 0 );

	// Distributed top-k: each rank tops its own shard, then the merge must reproduce a top-k over
	// the concatenated vocabulary exactly. Same tie-break independence as above.
	test_rmsnorm_residual_alias( arena, 8, 5120, false );
	test_rmsnorm_residual_alias( arena, 8, 5120, true );
	test_rmsnorm_residual_alias( arena, 3, 64, true );

	test_conv_side_split( arena, 8, 2, 8 );
	test_conv_side_split( arena, 5, 3, 4 );
	test_conv_side_split( arena, 1, 2, 320 );

	test_topk_merge( arena, 4, 7, 16, 62080, 62208 );
	test_topk_merge( arena, 2, 3, 8, 1024, 1024 );
	test_topk_merge( arena, 1, 1, 4, 256, 256 );

	engine::core::arena_destroy( arena );
	for ( int d = 0; d < ENGINE_NUM_GPUS; ++d ) {
		cudaSetDevice( d );
		cudaDeviceReset();
	}
	std::printf( "========================================================\n" );
	std::printf( "[+] DFLASH2 BACKBONE BLOCK PARITY PASSED\n" );
	std::printf( "========================================================\n" );
	return 0;
}
