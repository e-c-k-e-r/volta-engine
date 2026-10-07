// End-to-end parity for DFlashSpeculator::forward_backbone against an FP32 CPU reference.
//
// The block suite (test_dflash2_blocks) verifies each kernel in isolation: dynconv, RMSNorm with
// residual split, the FP16 GEMV, Q/K norm + RoPE, attention. None of that touches the thing that
// actually broke: the order in which forward_backbone composes those kernels and which buffer each
// one reads. Commit 5724fb0 fixed a real bug of exactly that shape -- the post-attention conv read
// the concatenated head outputs instead of o_proj's result, invisible at tp_size 1 because both are
// [rows, hidden]. No block test could have caught it.
//
// This builds a small synthetic backbone with dense FP16 weights, runs the real forward_backbone,
// and compares every intermediate stage against a CPU reference written in the reference's order.

#include "engine/config.h"
#include "engine/core/arena.h"
#include "engine/core/tensor.h"
#include "engine/spec/dflash.h"
#include "engine/modules/embedding.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using engine::spec::DFlashTestAccess;

namespace {

uint32_t g_seed = 0x12345678u;

float rnd( float s ) {
	g_seed = g_seed * 1664525u + 1013904223u;
	return ( static_cast<float>( g_seed & 0xffffu ) / 32768.0f - 1.0f ) * s;
}

// One ulp of the FP16 output format at |v|, derived from the exponent.
//
// The previous version returned a constant 2^13 for every v above 2^-8 and 1.0 below it, so the
// "ULP" figure was really max_abs / 8192 for large outputs and max_abs for small ones: two different
// metrics stitched together. That masked a real 1.4M-ULP failure as a 2.10-ULP pass.
double ulp_at( float v ) {
	const double a = fabs( static_cast<double>( v ) );
	if ( !( a > 0.0 ) || !std::isfinite( a ) ) return 0.0;
	int e = 0;
	std::frexp( a, &e );   // a = m * 2^e with m in [0.5, 1)
	// FP16 keeps 10 mantissa bits, so one ulp in that binade is 2^(e - 1 - 10).
	return std::ldexp( 1.0, e - 11 );
}

struct Diff {
	double max_abs = 0.0;
	double rms = 0.0;
	double max_ulp = 0.0;
};

// max_ulp is measured in ULPs of the tensor's largest value, not of each element.
//
// The output of a five-layer FP16 stack straddles zero. An element that happens to land at 1e-4 has
// an FP16 ulp of ~6e-8, so an ordinary absolute error of 7e-3 scores 1.2e5 "ULP" on that element while
// scoring 7.5 on an element of size 1. Per-element ULP distance therefore measures where the tensor
// happens to be small, not what the kernel computed. Scaling the denominator by the tensor's range
// expresses what is actually wanted: error relative to the signal.
//
// max_abs and rms are still computed against the unrounded reference and reported alongside, so
// nothing is hidden by this choice.
Diff compare( const std::vector<float>& ref, const std::vector<half>& got ) {
	Diff d;
	double ss = 0.0;
	const size_t n = ref.size();
	ENGINE_ASSERT( n == got.size(), "compare: reference and result sizes differ" );

	double scale = 0.0;
	for ( size_t i = 0; i < n; ++i ) scale = fmax( scale, fabs( static_cast<double>( ref[i] ) ) );
	if ( !( scale > 0.0 ) ) scale = 1.0;
	const double u_scale = ulp_at( static_cast<float>( scale ) );

	for ( size_t i = 0; i < n; ++i ) {
		const float g = __half2float( got[i] );
		const double r = ref[i];
		// A NaN makes every comparison false, which silently turns the whole check into a no-op
		// and reports a clean pass. Fail loudly instead.
		if ( !( std::isfinite( g ) && std::isfinite( r ) ) ) {
			std::fprintf( stderr, "[non-finite] i=%zu got=%.6g ref=%.6g\n", i, static_cast<double>( g ), r );
			ENGINE_ASSERT( false, "compare: non-finite value in reference or result" );
		}
		const double e = fabs( static_cast<double>( g ) - r );
		if ( e > d.max_abs ) d.max_abs = e;
		ss += e * e;
		// Distance to the FP16 rounding of the reference: the output is FP16, so anything under half
		// an ulp of the signal is exact and anything above is a real kernel error.
		const double rounded = __half2float( __float2half_rn( static_cast<float>( r ) ) );
		const double eq = fabs( static_cast<double>( g ) - rounded );
		d.max_ulp = fmax( d.max_ulp, eq / u_scale );
	}
	d.rms = ( n > 0 ) ? std::sqrt( ss / n ) : 0.0;
	return d;
}
// ---- FP32 CPU reference for one layer, in the reference's order ----

struct RefCfg {
	int H, rows, qh, kvh, hd, inter, taps, groups, gs;
};

// rmsnorm with residual split. residual_in may be null (first layer).
// BF16 round-to-nearest-even, keeping the value in FP32. Mirrors DFlashSM70's kernel.
static float ref_bf16_rne( float v ) {
	uint32_t b;
	std::memcpy( &b, &v, sizeof( b ) );
	b = ( b + 0x7FFFu + ( ( b >> 16 ) & 1u ) ) & 0xFFFF0000u;
	float o;
	std::memcpy( &o, &b, sizeof( o ) );
	return o;
}

void ref_rmsnorm_residual(
	const std::vector<half>& x,
	const std::vector<float>* rin,
	const std::vector<half>& w,
	float eps,
	std::vector<float>& residual_out,
	std::vector<half>& normed_out,
	int rows,
	int H
) {
	// Snapshot the residual input before writing the output. forward_backbone passes bb_residual_ as
	// both rin and residual_out at step (e), and this reference does the same at its call sites, so the
	// assign() below would zero the very buffer still being read as the input. That silently dropped
	// the incoming residual from every layer after the first, and it is why this reference disagreed
	// with a kernel that turned out to be exact.
	std::vector<float> rin_copy;
	if ( rin ) rin_copy = *rin;
	residual_out.assign( static_cast<size_t>( rows ) * H, 0.0f );
	normed_out.assign( static_cast<size_t>( rows ) * H, __float2half( 0.0f ) );
	for ( int t = 0; t < rows; ++t ) {
		double ss = 0.0;
		for ( int c = 0; c < H; ++c ) {
			// BF16 round-to-nearest-even on the residual transport, as DFlashSM70RMSNorm does. The
			// drafter is a BF16 model and the reference emulates BF16 exactly this way on SM70.
			const float v = ref_bf16_rne( __half2float( x[static_cast<size_t>( t ) * H + c] )
				+ ( rin ? rin_copy[static_cast<size_t>( t ) * H + c] : 0.0f ) );
			residual_out[static_cast<size_t>( t ) * H + c] = v;
			ss += static_cast<double>( v ) * v;
		}
		const float inv = static_cast<float>( rsqrt( static_cast<float>( ss / H ) + eps ) );
		for ( int c = 0; c < H; ++c ) {
			const float v = residual_out[static_cast<size_t>( t ) * H + c];
			const float wv = ref_bf16_rne( __half2float( w[c] ) );
			normed_out[static_cast<size_t>( t ) * H + c] = __float2half_rn( ref_bf16_rne( v * wv * inv ) );
		}
	}
}

// Grouped dynamic conv, causal over taps, matching test_dynconv's reference.
void ref_dynconv(
	const std::vector<half>& x,
	const std::vector<float>& dyn,   // [rows, taps, groups] FP32 coefficients
	const std::vector<half>& base,   // [taps, H]
	const RefCfg& c,
	std::vector<half>& out
) {
	out.assign( static_cast<size_t>( c.rows ) * c.H, __float2half( 0.0f ) );
	for ( int t = 0; t < c.rows; ++t ) {
		for ( int ch = 0; ch < c.H; ++ch ) {
			const int g = ch / c.gs;
			double acc = 0.0;
			for ( int k = 0; k < c.taps; ++k ) {
				if ( k > t ) break;
				const double wv = __half2float( base[static_cast<size_t>( k ) * c.H + ch] )
					+ dyn[( ( static_cast<size_t>( t ) * c.taps ) + k ) * c.groups + g];
				acc += wv * __half2float( x[( static_cast<size_t>( t ) - k ) * c.H + ch] );
			}
			out[static_cast<size_t>( t ) * c.H + ch] = __float2half_rn( static_cast<float>( acc ) );
		}
	}
}

// Dense FP16 GEMV, [m, k] x [k, n] -> [m, n].


void ref_gemv(
	const std::vector<half>& x,
	const std::vector<half>& w,
	int m,
	int n,
	int k,
	std::vector<half>& out
) {
	// The weight is [n, k] row-major, not [k, n]: dflash_f16_gemv_nt_kernel walks wcol = w + col * k,
	// so each output column's k values are contiguous. The previous version indexed w[i * n + j] and
	// was computing a transposed product -- plausible magnitudes, wrong answer, and it survived two
	// green runs because the ULP metric above was broken at the same time.
	//
	// Accumulation is float, matching the kernel's fmaf chain.
	out.assign( static_cast<size_t>( m ) * n, __float2half( 0.0f ) );
	for ( int r = 0; r < m; ++r ) {
		for ( int j = 0; j < n; ++j ) {
			float acc = 0.0f;
			for ( int i = 0; i < k; ++i )
				acc += __half2float( w[static_cast<size_t>( j ) * k + i] )
					* __half2float( x[static_cast<size_t>( r ) * k + i] );
			out[static_cast<size_t>( r ) * n + j] = __float2half_rn( acc );
		}
	}
}

void ref_swiglu( const std::vector<half>& gate, const std::vector<half>& up, int m, int n, std::vector<half>& out ) {
	out.assign( static_cast<size_t>( m ) * n, __float2half( 0.0f ) );
	for ( size_t i = 0; i < static_cast<size_t>( m ) * n; ++i ) {
		const float g = __half2float( gate[i] );
		const float u = __half2float( up[i] );
		out[i] = __float2half_rn( ( g / ( 1.0f + std::exp( -g ) ) ) * u );
	}
}

// Per-head RMS norm with a learned weight, then NeoX (rotate-half) RoPE over the first rope_dims
// components. Q and K are normalized independently, each with its own head count.
void ref_qk_norm_rope(
	const std::vector<half>& x,
	const std::vector<half>& w,
	int rows,
	int heads,
	int hd,
	int rope_pairs,
	int rope_dims,
	float theta,
	int pos0,
	const std::vector<float>& cos_t,
	const std::vector<float>& sin_t,
	std::vector<half>& out
) {
	const float eps = 1e-6f;
// Snapshot x when the caller aliases it as the output (the engine normalizes in place).
	std::vector<half> x_in;
	if ( &out == &x ) x_in = x;
	const std::vector<half>& xs = ( &out == &x ) ? x_in : x;
	out.assign( xs.size(), __float2half( 0.0f ) );
	for ( int r = 0; r < rows; ++r ) {
		for ( int h = 0; h < heads; ++h ) {
			const size_t base = ( static_cast<size_t>( r ) * heads + h ) * hd;
			double ss = 0.0;
			for ( int d = 0; d < hd; ++d ) {
				const double v = __half2float( xs[base + d] );
				ss += v * v;
			}
			const float inv = rsqrtf( static_cast<float>( ss / hd ) + eps );
			std::vector<float> n( hd );
			for ( int d = 0; d < hd; ++d )
				n[d] = __half2float( xs[base + d] ) * inv * __half2float( w[d] );
			const int pos = pos0 + r;
			for ( int d = 0; d < hd; ++d ) {
				if ( d < rope_dims ) {
					const int p = ( d < rope_pairs ) ? d : d - rope_pairs;
					const float c = cos_t[static_cast<size_t>( pos ) * rope_pairs + p];
					const float si = sin_t[static_cast<size_t>( pos ) * rope_pairs + p];
					if ( d < rope_pairs ) out[base + d] = __float2half_rn( n[d] * c - n[d + rope_pairs] * si );
					else out[base + d] = __float2half_rn( n[d] * c + n[d - rope_pairs] * si );
				}
				else out[base + d] = __float2half_rn( n[d] );
			}
		}
	}
}

// Non-causal over the block's own rows, matching dflash_attn_kernel. The block contributes rows
// 1..rows-1, not 0..rows-1: row 0 is the anchor, whose position is covered by the context side, and
// upstream's tap write overwrites the transient row-0 K/V there. With n_ctx == 0 (this test's case)
// that leaves rows 1..rows-1 as the whole key set. The drafter's config.json has is_causal = False for
// all five layers, so every drafter layer attends over the whole block. An earlier version of this
// reference was causal (t <= r), which is a different function.
void ref_attn(
	const std::vector<half>& q,
	const std::vector<half>& k,
	const std::vector<half>& v,
	int rows,
	int qh,
	int kvh,
	int hd,
	std::vector<half>& out
) {
	const int rep = qh / kvh;
	const float scale = rsqrtf( static_cast<float>( hd ) );
	// Key t is block row t + 1.
	const int nblk = rows - 1;
	out.assign( static_cast<size_t>( rows ) * qh * hd, __float2half( 0.0f ) );
	// No context: every key comes from the block's own K/V.
	for ( int r = 0; r < rows; ++r ) {
		for ( int h = 0; h < qh; ++h ) {
			const int kvh_i = h / rep;
			std::vector<float> scores( nblk );
			float mx = -INFINITY;
			for ( int t = 0; t < nblk; ++t ) {
				double acc = 0.0;
				for ( int i = 0; i < hd; ++i ) {
					acc += static_cast<double>( __half2float( q[( static_cast<size_t>( r ) * qh + h ) * hd + i] ) )
						* __half2float( k[( static_cast<size_t>( t + 1 ) * kvh + kvh_i ) * hd + i] );
				}
				scores[t] = static_cast<float>( acc ) * scale;
				if ( scores[t] > mx ) mx = scores[t];
			}
			double sum = 0.0;
			std::vector<float> p( nblk );
			for ( int t = 0; t < nblk; ++t ) {
				p[t] = std::exp( static_cast<double>( scores[t] - mx ) );
				sum += p[t];
			}
			for ( int i = 0; i < hd; ++i ) {
				double acc = 0.0;
				for ( int t = 0; t < nblk; ++t ) {
					const double w = p[t] / sum;
					acc += w * __half2float( v[( static_cast<size_t>( t + 1 ) * kvh + kvh_i ) * hd + i] );
				}
				out[( static_cast<size_t>( r ) * qh + h ) * hd + i] = __float2half_rn( static_cast<float>( acc ) );
			}
		}
	}
}

// Attention with the projected target context in front of the block's own K/V. Keys 0..n_ctx-1 are the
// context positions 0..n_ctx-1 of the layer, keys n_ctx.. are block rows 1..rows-1, mirroring the
// kernel's two key ranges. This is the n_ctx > 0 path, which had no reference at all: every parity run
// so far passed n_ctx = 0, so the fc projection, hidden_norm, the context k/v projection, its k_norm,
// its RoPE and the ring indexing were all unchecked.
void ref_attn_ctx(
	const std::vector<half>& q,
	const std::vector<half>& ck,
	const std::vector<half>& cv,
	const std::vector<half>& bk,
	const std::vector<half>& bv,
	int rows,
	int qh,
	int kvh,
	int hd,
	int n_ctx,
	std::vector<half>& out
) {
	const int rep = qh / kvh;
	const float scale = rsqrtf( static_cast<float>( hd ) );
	const int nkeys = n_ctx + rows - 1;
	out.assign( static_cast<size_t>( rows ) * qh * hd, __float2half( 0.0f ) );
	for ( int r = 0; r < rows; ++r ) {
		for ( int h = 0; h < qh; ++h ) {
			const int g = h / rep;
			auto key_at = [&]( int kk ) {
				return kk < n_ctx
					? ( static_cast<size_t>( kk ) * kvh + g ) * hd
					: ( static_cast<size_t>( kk - n_ctx + 1 ) * kvh + g ) * hd;
			};
			std::vector<float> scores( nkeys );
			float mx = -INFINITY;
			for ( int kk = 0; kk < nkeys; ++kk ) {
				const size_t kb = key_at( kk );
				double acc = 0.0;
				for ( int i = 0; i < hd; ++i )
					acc += static_cast<double>( __half2float( q[( static_cast<size_t>( r ) * qh + h ) * hd + i] ) )
						* __half2float( ( kk < n_ctx ? ck : bk )[kb + i] );
				scores[kk] = static_cast<float>( acc ) * scale;
				mx = fmaxf( mx, scores[kk] );
			}
			double sum = 0.0;
			std::vector<float> p( nkeys );
			for ( int kk = 0; kk < nkeys; ++kk ) {
				p[kk] = std::exp( static_cast<double>( scores[kk] - mx ) );
				sum += p[kk];
			}
			for ( int i = 0; i < hd; ++i ) {
				double acc = 0.0;
				for ( int kk = 0; kk < nkeys; ++kk )
					acc += ( p[kk] / sum ) * __half2float( ( kk < n_ctx ? cv : bv )[key_at( kk ) + i] );
				out[( static_cast<size_t>( r ) * qh + h ) * hd + i] = __float2half_rn( static_cast<float>( acc ) );
			}
		}
	}
}

struct Synthetic {
	pod::DFlashConfig cfg;
	std::vector<engine::spec::DFlashLayer> layers;
	pod::TensorView final_norm;

	// Host weight data, kept so the reference can read the same values.
	std::vector<std::vector<half>> wq, wk, wv, wo, wgate, wup, wdown, wdyn_a, wdyn_m, base_a, base_m;
	std::vector<std::vector<half>> norm_in, norm_post;
	std::vector<half> final_w;
	std::vector<half> qnorm_w, knorm_w;
	std::vector<half> embed;
	// Context stage: `fc` projects the 5 concatenated target taps (5*H) down to H, `hidden_norm` is the
	// RMSNorm that follows it. Both are needed to drive the n_ctx > 0 path with weights the CPU can
	// evaluate; without them forward_backbone skips the context entirely.
	std::vector<half> wfc;
	std::vector<half> hnorm_w;

	int H, rows, qh, kvh, hd, inter, taps, groups, gs, proj_out, rope_pairs;
};

pod::TensorView alloc( pod::MemoryArena& arena, pod::DType dt, const int* shape, int ndim ) {
	return engine::core::arena_alloc_tensor( arena, dt, shape, ndim );
}

void upload( const std::vector<half>& h, pod::TensorView& t ) {
	CUDA_CHECK( cudaMemcpy( t.data, h.data(), h.size() * 2, cudaMemcpyHostToDevice ) );
}

Synthetic build( int H_, int L, int rows_, int qh, int kvh, int hd, int inter, int gs = 8, int vocab = 512 ) {
	Synthetic s;
	s.H = H_; s.rows = rows_; s.qh = qh; s.kvh = kvh; s.hd = hd; s.inter = inter;
	s.taps = 2; s.gs = gs; s.groups = H_ / s.gs; s.proj_out = 2 * s.taps * s.groups; s.rope_pairs = hd / 2;

	pod::DFlashConfig& c = s.cfg;
	c.hidden_size = H_;
	c.num_hidden_layers = L;
	c.block_size = rows_;
	c.num_q_heads = qh;
	c.num_kv_heads = kvh;
	c.head_dim = hd;
	c.intermediate_size = inter;
	c.conv_kernel_size = s.taps;
	c.conv_group_size = s.gs;
	c.selector_rank = 16;
	c.selector_top_k = 16;
	c.vocab_size = vocab;
	c.sliding_window = 32;
	c.rms_norm_eps = 1e-6f;
	c.rope_theta = 1.0e4f;
	c.rope_dim_sections[0] = hd;
	c.num_hidden_layers = L;
	c.num_target_layer_ids = 1;

	s.layers.resize( L );
	s.wq.resize( L ); s.wk.resize( L ); s.wv.resize( L ); s.wo.resize( L );
	s.wgate.resize( L ); s.wup.resize( L ); s.wdown.resize( L );
	s.wdyn_a.resize( L ); s.wdyn_m.resize( L ); s.base_a.resize( L ); s.base_m.resize( L );
	s.norm_in.resize( L ); s.norm_post.resize( L );

	auto fill = []( std::vector<half>& v, size_t n, float s_ ) {
		v.resize( n );
		for ( auto& x : v ) x = __float2half_rn( rnd( s_ ) );
	};
	auto fill_pos = []( std::vector<half>& v, size_t n ) {
		v.resize( n );
		for ( auto& x : v ) x = __float2half_rn( 0.5f + rnd( 0.5f ) );
	};

	s.embed.resize( static_cast<size_t>( vocab ) * H_ );
	for ( auto& x : s.embed ) x = __float2half_rn( rnd( 1.0f ) );

	// Weight scale is normalised by 1/sqrt(k) so that a unit-variance activation produces an
	// O(1) output at any contraction length. Without it the synthetic stack grows like sqrt(k) per
	// projection and saturates FP16 outright at the production width: post_attn_conv reached 63136
	// against an FP16 max of 65504 and the residual overflowed to inf.
	// Only normalise at production width. At hidden=64 the unscaled synthetic weights already keep
	// the final output in FP16's normal range; normalising there as well drives the output into
	// subnormals (min normal 6.1e-5), where the output format's own precision collapses and the
	// comparison measures the format rather than the kernel.
	const bool norm_w = ( H_ > 256 );
	const auto wscale = [&]( float sigma, int k ) { return norm_w ? sigma / sqrtf( (float) k ) : sigma; };

	for ( int l = 0; l < L; ++l ) {
		fill( s.wq[l], static_cast<size_t>( H_ ) * qh * hd, wscale( 0.20f, H_ ) );
		fill( s.wk[l], static_cast<size_t>( H_ ) * kvh * hd, wscale( 0.20f, H_ ) );
		fill( s.wv[l], static_cast<size_t>( H_ ) * kvh * hd, wscale( 0.20f, H_ ) );
		fill( s.wo[l], static_cast<size_t>( qh * hd ) * H_, wscale( 0.06f, qh * hd ) );
		fill( s.wgate[l], static_cast<size_t>( H_ ) * inter, wscale( 0.15f, H_ ) );
		fill( s.wup[l], static_cast<size_t>( H_ ) * inter, wscale( 0.15f, H_ ) );
		fill( s.wdown[l], static_cast<size_t>( inter ) * H_, wscale( 0.04f, inter ) );
		fill_pos( s.norm_in[l], static_cast<size_t>( H_ ) );
		fill_pos( s.norm_post[l], static_cast<size_t>( H_ ) );
		fill( s.wdyn_a[l], static_cast<size_t>( s.proj_out ) * H_, wscale( 0.10f, H_ ) );
		fill( s.wdyn_m[l], static_cast<size_t>( s.proj_out ) * H_, wscale( 0.10f, H_ ) );
		fill( s.base_a[l], static_cast<size_t>( 2 * s.taps ) * H_, 0.30f );
		fill( s.base_m[l], static_cast<size_t>( 2 * s.taps ) * H_, 0.30f );
	}
	fill_pos( s.final_w, static_cast<size_t>( H_ ) );
	// fc is [H, 5*H]: one row per hidden channel, contracting the concatenated taps. The scale has to
	// account for the 5x wider contraction or the projected context leaves FP16's normal range.
	fill( s.wfc, static_cast<size_t>( H_ ) * 5 * H_, wscale( 0.20f, 5 * H_ ) );
	fill_pos( s.hnorm_w, static_cast<size_t>( H_ ) );
	for ( int i = 0; i < hd; ++i ) {
		s.qnorm_w.push_back( __float2half_rn( 0.8f + rnd( 0.2f ) ) );
		s.knorm_w.push_back( __float2half_rn( 0.8f + rnd( 0.2f ) ) );
	}
	return s;
}

} // namespace

int main( int argc, char** argv ) {
	std::printf( "========================================================\n" );
	std::printf( "   volta-engine Tier 2: DFlash2 end-to-end backbone    \n" );
	std::printf( "========================================================\n" );

	pod::MemoryArena arena;
	engine::spec::DFlashSpeculator spec;

	// qh * head_dim != hidden, so a [n, k] vs [k, n] weight-layout error cannot hide behind a
	// square shape the way it could at qh = 4 / head_dim = 16 / hidden = 64.
	// argv[1] selects the geometry. "small" is the default; "real" uses the production attention
	// shape (hidden 5120, 32 q heads, 8 kv heads, head_dim 128, conv group 16 -> 320 groups) with a
	// reduced MLP width so the CPU reference stays quick. The attention and conv geometry is the part
	// where head count, GQA ratio and group count change kernel indexing, so that is what this covers.
	int H = 64, L = 2, QH = 8, KVH = 2, HD = 16, INTER = 96, GS = 8, VOCAB = 512;
	if ( argc > 1 && std::string( argv[1] ) == "real" ) {
		H = 5120; L = 1; QH = 32; KVH = 8; HD = 128; INTER = 256; GS = 16; VOCAB = 512;
		std::printf( "  [geometry] production attention shape: hidden=%d layers=%d qh=%d kvh=%d hd=%d gs=%d groups=%d\n",
			H, L, QH, KVH, HD, GS, H / GS );
	}
	if ( argc > 1 && std::string( argv[1] ) == "glm" ) {
		H = 4096; L = 1; QH = 32; KVH = 8; HD = 128; INTER = 256; GS = 16; VOCAB = 512;
		std::printf( "  [geometry] GLM drafter attention shape: hidden=%d layers=%d qh=%d kvh=%d hd=%d gs=%d groups=%d\n",
			H, L, QH, KVH, HD, GS, H / GS );
	}
	const int ROWS = 8;

	Synthetic s = build( H, L, ROWS, QH, KVH, HD, INTER, GS, VOCAB );
	CUDA_CHECK( cudaSetDevice( 0 ) );
	// Production geometry needs more: the RoPE table alone is rope_len x head_dim/2 x fp32 for both
	// cos and sin, and the context stage's buffers are ctx_chunk rows wide.
	engine::core::arena_create( arena, 0, ( H > 256 ) ? 4ULL * 1024 * 1024 * 1024 : 512ULL * 1024 * 1024 );

	if ( !spec.init( s.cfg, 0, arena, nullptr, 0, 1 ) ) {
		std::printf( "[FATAL] speculator init failed\n" );
		return 1;
	}

	// Install dense FP16 weights.
	{
		std::vector<half> final_w = s.final_w;
		int nw[1] = { H };
		pod::TensorView fn = alloc( arena, pod::DType::FP16, nw, 1 );
		upload( final_w, fn );

		for ( int l = 0; l < L; ++l ) {
			engine::spec::DFlashLayer& ly = s.layers[l];
			auto put = [&]( std::vector<half>& data, int k, int n ) {
				int sh[2] = { k, n };
				pod::TensorView t = alloc( arena, pod::DType::FP16, sh, 2 );
				upload( data, t );
				return pod::LinearWeight::make_fp16( t );
			};
			ly.q_proj = put( s.wq[l], H, QH * HD );
			ly.k_proj = put( s.wk[l], H, KVH * HD );
			ly.v_proj = put( s.wv[l], H, KVH * HD );
			ly.o_proj = put( s.wo[l], QH * HD, H );
			ly.gate_proj = put( s.wgate[l], H, INTER );
			ly.up_proj = put( s.wup[l], H, INTER );
			ly.down_proj = put( s.wdown[l], INTER, H );

			int nw1[1] = { H };
			pod::TensorView ni = alloc( arena, pod::DType::FP16, nw1, 1 );
			pod::TensorView np = alloc( arena, pod::DType::FP16, nw1, 1 );
			upload( s.norm_in[l], ni );
			upload( s.norm_post[l], np );
			ly.input_norm = ni;
			ly.post_attn_norm = np;

			int ap[2] = { s.proj_out, H };
			pod::TensorView apj = alloc( arena, pod::DType::FP16, ap, 2 );
			pod::TensorView mpj = alloc( arena, pod::DType::FP16, ap, 2 );
			upload( s.wdyn_a[l], apj );
			upload( s.wdyn_m[l], mpj );
			ly.attn_conv_proj = apj;
			ly.mlp_conv_proj = mpj;

			int ab[3] = { 2, s.taps, H };
			pod::TensorView abt = alloc( arena, pod::DType::FP16, ab, 3 );
			pod::TensorView mbt = alloc( arena, pod::DType::FP16, ab, 3 );
			upload( s.base_a[l], abt );
			upload( s.base_m[l], mbt );
			ly.attn_conv_base = abt;
			ly.mlp_conv_base = mbt;

			int qd[1] = { HD };
			pod::TensorView qn = alloc( arena, pod::DType::FP16, qd, 1 );
			pod::TensorView kn = alloc( arena, pod::DType::FP16, qd, 1 );
			upload( s.qnorm_w, qn );
			upload( s.knorm_w, kn );
			ly.q_norm = qn;
			ly.k_norm = kn;
		}
		DFlashTestAccess::set_dense_weights( spec, s.layers.data(), fn );

		// fc (5*H -> H) and hidden_norm drive the context projection. Without them forward_backbone's
		// `n_ctx > 0 && fc_.is_valid()` guard skips the context and the n_ctx > 0 path runs untested.
		{
			int fcs[2] = { 5 * H, H };
			pod::TensorView fct = alloc( arena, pod::DType::FP16, fcs, 2 );
			upload( s.wfc, fct );
			int hn[1] = { H };
			pod::TensorView hnt = alloc( arena, pod::DType::FP16, hn, 1 );
			upload( s.hnorm_w, hnt );
			DFlashTestAccess::set_context_weights( spec, pod::LinearWeight::make_fp16( fct ), hnt );
		}
	}

	// Input ids and the embedding table on device.
	{
// The embedding table is BF16 (the target's checkpoint dtype); the gather emits FP16.
	int es[2] = { VOCAB, H };
	pod::TensorView et = alloc( arena, pod::DType::BF16, es, 2 );
	{
		std::vector<__nv_bfloat16> eb( s.embed.size() );
		for ( size_t i = 0; i < eb.size(); ++i ) eb[i] = __float2bfloat16( __half2float( s.embed[i] ) );
		CUDA_CHECK( cudaMemcpy( et.data, eb.data(), eb.size() * 2, cudaMemcpyHostToDevice ) );
	}
		std::vector<int> ids( ROWS );
		for ( int i = 0; i < ROWS; ++i ) ids[i] = ( i * 37 + 11 ) % VOCAB;
		CUDA_CHECK( cudaMemcpy( DFlashTestAccess::input_ids( spec ), ids.data(), ROWS * sizeof( int ), cudaMemcpyHostToDevice ) );
		DFlashTestAccess::set_embed( spec, et );
	}

	int hs[2] = { ROWS, H };
	pod::TensorView hidden_out = alloc( arena, pod::DType::FP16, hs, 2 );

	// The block sits at BLK_POS0..BLK_POS0+ROWS-1. With context on, its row 0 is the anchor, whose
	// absolute position is the last context position -- upstream's tap write replaces the block's
	// row-0 K/V there -- so the two ranges overlap by exactly that one position. The block no longer
	// starts at 0, so this also checks that RoPE and the key set follow the absolute position rather
	// than a row index.
	const int CTX_N = 12, CTX_POS0 = 20, BLK_POS0 = CTX_POS0 + CTX_N - 1;
	std::vector<half> taps( static_cast<size_t>( CTX_N ) * 5 * H );
	for ( auto& x : taps ) x = __float2half_rn( rnd( 1.0f ) );
	// Device memory: forward_backbone reads the taps on the GPU, and the engine hands it the target's
	// device-side tap ring.
	int ts[2] = { CTX_N, 5 * H };
	pod::TensorView taps_dev = alloc( arena, pod::DType::FP16, ts, 2 );
	upload( taps, taps_dev );

	// n_ctx = 0 runs first: forward_backbone advances kv_len_ by n_ctx, so the context case has to
	// start its ring from 0.
	auto run_gpu = [&]( bool use_ctx ) {
		spec.forward_backbone(
			DFlashTestAccess::embed( spec ), DFlashTestAccess::input_ids( spec ), ROWS,
			use_ctx ? taps_dev.ptr<const half>() : nullptr, use_ctx ? CTX_N : 0, CTX_POS0, BLK_POS0, hidden_out, nullptr
		);
		CUDA_CHECK( cudaDeviceSynchronize() );
		std::vector<half> o( static_cast<size_t>( ROWS ) * H );
		CUDA_CHECK( cudaMemcpy( o.data(), hidden_out.data, o.size() * 2, cudaMemcpyDeviceToHost ) );
		return o;
	};

	// ---- CPU reference ----
	// Layer 0's context K, so a failure can be attributed to the context projection rather than to the
	// attention that consumes it.
	std::vector<half> ref_ctx_k;
	auto compute_ref = [&]( bool use_ctx ) -> std::vector<float> {
	RefCfg cf{ H, ROWS, QH, KVH, HD, INTER, s.taps, s.groups, s.gs };

	std::vector<int> ids( ROWS );
	for ( int i = 0; i < ROWS; ++i ) ids[i] = ( i * 37 + 11 ) % VOCAB;
	std::vector<half> stage( static_cast<size_t>( ROWS ) * H );
	for ( int r = 0; r < ROWS; ++r )
		for ( int c = 0; c < H; ++c )
			stage[static_cast<size_t>( r ) * H + c] = s.embed[static_cast<size_t>( ids[r] ) * H + c];

	std::vector<float> residual( static_cast<size_t>( ROWS ) * H, 0.0f );
	std::vector<half> normed( static_cast<size_t>( ROWS ) * H );
	std::vector<half> tmp, q, k, v, qo, ko, attn, o, gate, up, mlp, dyn, stage2;

	// RoPE tables for an absolute position range.
	auto rope_tables = [&]( int pos0, int n ) {
		std::vector<float> c( static_cast<size_t>( n ) * s.rope_pairs );
		std::vector<float> sn( static_cast<size_t>( n ) * s.rope_pairs );
		for ( int r = 0; r < n; ++r )
			for ( int p = 0; p < s.rope_pairs; ++p ) {
				const float inv = 1.0f / powf( s.cfg.rope_theta, ( 2.0f * p ) / HD );
				c[static_cast<size_t>( r ) * s.rope_pairs + p] = std::cos( ( pos0 + r ) * inv );
				sn[static_cast<size_t>( r ) * s.rope_pairs + p] = std::sin( ( pos0 + r ) * inv );
			}
		return std::make_pair( c, sn );
	};

	// The block sits at BLK_POS0..BLK_POS0+ROWS-1 (declared above, next to run_gpu).
	const auto ctx_tables = rope_tables( CTX_POS0, CTX_N );
	const auto blk_tables = rope_tables( BLK_POS0, ROWS );

	auto coef_from = [&]( const std::vector<half>& proj, int side ) {
		// proj is [rows, 2*taps*groups]; side 0 is the first taps*groups, side 1 the second.
		std::vector<float> cf2( static_cast<size_t>( ROWS ) * s.taps * s.groups );
		const int blk = s.taps * s.groups;
		for ( int r = 0; r < ROWS; ++r )
			for ( int i = 0; i < blk; ++i )
				cf2[static_cast<size_t>( r ) * blk + i] =
					__half2float( proj[static_cast<size_t>( r ) * ( 2 * blk ) + side * blk + i] );
		return cf2;
	};

	// Context projection: concat the 5 taps, project with fc, RMSNorm with hidden_norm. Per layer the
	// K/V come off that projected stream, k is k_norm'd and RoPE'd at the token's absolute position,
	// and V is stored unrotated -- the same order dflash_update_kv_from_target uses.
	std::vector<half> ctx_normed;
	std::vector<half> ctx_fc;
	{
		std::vector<float> resid( static_cast<size_t>( CTX_N ) * H );
		ref_gemv( taps, s.wfc, CTX_N, H, 5 * H, ctx_fc );
		ref_rmsnorm_residual( ctx_fc, nullptr, s.hnorm_w, s.cfg.rms_norm_eps, resid, ctx_normed, CTX_N, H );
	}
	std::vector<std::vector<half>> ctx_k( L ), ctx_v( L );

	for ( int l = 0; l < L; ++l ) {
		ref_rmsnorm_residual( stage, ( l == 0 ) ? nullptr : &residual, s.norm_in[l], s.cfg.rms_norm_eps, residual, normed, ROWS, H );

		ref_gemv( normed, s.wdyn_a[l], ROWS, s.proj_out, H, dyn );
		ref_dynconv( normed, coef_from( dyn, 0 ), s.base_a[l], cf, stage2 );

		ref_gemv( stage2, s.wq[l], ROWS, QH * HD, H, q );
		ref_gemv( stage2, s.wk[l], ROWS, KVH * HD, H, k );
		ref_gemv( stage2, s.wv[l], ROWS, KVH * HD, H, v );
		ref_qk_norm_rope( q, s.qnorm_w, ROWS, QH, HD, s.rope_pairs, HD, s.cfg.rope_theta, 0, blk_tables.first, blk_tables.second, qo );
		ref_qk_norm_rope( k, s.knorm_w, ROWS, KVH, HD, s.rope_pairs, HD, s.cfg.rope_theta, 0, blk_tables.first, blk_tables.second, ko );
		if ( use_ctx ) {
			ref_gemv( ctx_normed, s.wk[l], CTX_N, KVH * HD, H, ctx_k[l] );
			ref_gemv( ctx_normed, s.wv[l], CTX_N, KVH * HD, H, ctx_v[l] );
			ref_qk_norm_rope( ctx_k[l], s.knorm_w, CTX_N, KVH, HD, s.rope_pairs, HD, s.cfg.rope_theta, 0,
				ctx_tables.first, ctx_tables.second, ctx_k[l] );
			ref_attn_ctx( qo, ctx_k[l], ctx_v[l], ko, v, ROWS, QH, KVH, HD, CTX_N, attn );
		}
		else {
			ref_attn( qo, ko, v, ROWS, QH, KVH, HD, attn );
		}
		ref_gemv( attn, s.wo[l], ROWS, H, QH * HD, o );

		// (d) post-attention conv consumes o_proj's output.
		ref_dynconv( o, coef_from( dyn, 1 ), std::vector<half>( s.base_a[l].begin() + s.taps * H, s.base_a[l].end() ), cf, stage2 );

		ref_rmsnorm_residual( stage2, &residual, s.norm_post[l], s.cfg.rms_norm_eps, residual, normed, ROWS, H );

		ref_gemv( normed, s.wdyn_m[l], ROWS, s.proj_out, H, dyn );
		ref_dynconv( normed, coef_from( dyn, 0 ), s.base_m[l], cf, stage2 );
		ref_gemv( stage2, s.wgate[l], ROWS, INTER, H, gate );
		ref_gemv( stage2, s.wup[l], ROWS, INTER, H, up );
			ref_swiglu( gate, up, ROWS, INTER, mlp );
		ref_gemv( mlp, s.wdown[l], ROWS, H, INTER, o );
		ref_dynconv( o, coef_from( dyn, 1 ), std::vector<half>( s.base_m[l].begin() + s.taps * H, s.base_m[l].end() ), cf, stage2 );

		stage = stage2;
	}

	std::vector<float> final_res;
	std::vector<half> final_normed;
	// The reference finishes the stack with `self.norm(hidden_states, residual)`, so the accumulated
	// residual is part of the final norm's input. Normalising the last sublayer output alone was wrong
	// and this gate agreed with the bug because the reference chain mirrored it.
	ref_rmsnorm_residual( stage, &residual, s.final_w, s.cfg.rms_norm_eps, final_res, final_normed, ROWS, H );

std::vector<float> ref( final_normed.size() );
	for ( size_t i = 0; i < ref.size(); ++i ) ref[i] = __half2float( final_normed[i] );
	ref_ctx_k = ctx_k[0];
	return ref;
	};

	// Both cases: the block alone, and the block over a projected target context. The second one is the
	// path the engine actually runs in decode, and until now nothing checked it against a reference.
	for ( int pass = 0; pass < 2; ++pass ) {
		const bool use_ctx = pass == 1;
		const std::vector<half> h_out = run_gpu( use_ctx );
		const std::vector<float> ref = compute_ref( use_ctx );
		if ( use_ctx ) {
			// The context K/V the GPU actually cached, layer 0, slots 0..CTX_N-1, against the
			// reference's fc -> hidden_norm -> k_proj -> k_norm -> RoPE chain.
			const int L2 = L, kvh = KVH, hd = HD, stride_slot = L2 * kvh * hd;
			const size_t n = static_cast<size_t>( CTX_N ) * kvh * hd;
			// The gather strides by whole cache slots, so copy the strided extent, not the packed one.
			const size_t raw = static_cast<size_t>( CTX_N ) * stride_slot;
			std::vector<half> gk( raw );
			CUDA_CHECK( cudaMemcpy( gk.data(), DFlashTestAccess::cache_k( spec ), gk.size() * 2, cudaMemcpyDeviceToHost ) );
			// Gather the context slots out of the [kv_total, L, kvh, hd] cache.
			std::vector<half> gg( static_cast<size_t>( CTX_N ) * kvh * hd );
			for ( int s = 0; s < CTX_N; ++s )
				for ( int i = 0; i < kvh * hd; ++i )
					gg[static_cast<size_t>( s ) * kvh * hd + i] = gk[static_cast<size_t>( s ) * stride_slot + i];
			int nbad = 0;
			for ( size_t z = 0; z < gg.size(); ++z )
				if ( !std::isfinite( __half2float( gg[z] ) ) ) ++nbad;
			std::printf( "  ctxK      nonfinite=%d/%zu\n", nbad, gg.size() );
			std::vector<float> rck( ref_ctx_k.size() );
			for ( size_t z = 0; z < rck.size(); ++z ) rck[z] = __half2float( ref_ctx_k[z] );
			const Diff dc = compare( rck, gg );
			std::printf( "  ctxK      n=%d layer=0 | max %.3e rms %.3e ulp %.2f\n", CTX_N, dc.max_abs, dc.rms, dc.max_ulp );
			ENGINE_ASSERT( dc.max_ulp <= 16.0, "context K deviates from the FP32 reference" );
			std::fflush( stdout );
		}
		const Diff d = compare( ref, h_out );
		std::printf( "  backbone  n_ctx=%-4d rows=%d hidden=%d layers=%d qh=%d kvh=%d hd=%d inter=%d | max %.3e rms %.3e ulp %.2f\n",
			use_ctx ? CTX_N : 0, ROWS, H, L, QH, KVH, HD, INTER, d.max_abs, d.rms, d.max_ulp );
		std::fflush( stdout );
		ENGINE_ASSERT( d.max_ulp <= 16.0,
			use_ctx ? "context-stage backbone deviates from the FP32 reference"
			        : "end-to-end backbone deviates from the FP32 reference" );
	}

	spec.destroy();
	engine::core::arena_destroy( arena );

	std::printf( "========================================================\n" );
	std::printf( "[+] DFLASH2 END-TO-END BACKBONE PARITY PASSED\n" );
	std::printf( "========================================================\n" );
	return 0;
}