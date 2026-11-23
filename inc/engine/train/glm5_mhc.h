#pragma once

// Training forms of GLM-5.3-Flash (glm5_next) operations, FP32 on the caller's stream (see .agents/GLM_STAGEC_PLAN.md): the mHC hyper-connection
// site, its pre step before each sublayer, its post step after it and the head's stream mean. Each forward keeps what its backward needs, each
// backward adds the input gradients into the caller's buffers (+=) unless it says otherwise. They follow the inference kernels' math
// (engine/modules/mhc.h) in plain FP32: expf, IEEE divisions and square roots, no FP16 roundings. The Sinkhorn backward is unrolled through
// every normalization of the forward, the eps terms included.
//
// Streams X are [rows, kHc, hidden] FP32, stream h of row r at ( r * kHc + h ) * hidden; a row's flat view x has x[h * hidden + d] = X_h[d].
// The pre step of a row:
//   mix = fn x (kHcMix logits), rmr = 1 / sqrt( |x|^2 / ( kHc hidden ) + rms_eps ), l = mix rmr
//   pre_h = sigmoid( l_h scale[0] + base[h] ) + hc_eps, post_h = 2 sigmoid( l_{4+h} scale[1] + base[4+h] )
//   comb: E[g][k] = l_{8+4g+k} scale[2] + base[8+4g+k]; V[g][:] = softmax( E[g][:] ) + hc_eps; one column normalization
//     V[:, k] /= ( sum_g V[g][k] + hc_eps ), then iters - 1 x ( row normalization V[g, :] /= ( sum_k V[g][k] + hc_eps ), column normalization );
//     comb = V (g-major)
//   u = sum_h pre_h X_h, a = u norm_w / sqrt( mean( u^2 ) + norm_eps ) (the sublayer's input)
// The post step, from the sublayer's output y [rows, hidden]: X_new_h = post_h y + sum_g comb[g][h] X_g.
#include "engine/train/grad.h"
#include <cstddef>

namespace engine::train::glm5_next {

constexpr int kHc = 4;                    // streams
constexpr int kHcMix = 24;                // mix logits per row: kHc pre, kHc post, kHc * kHc comb
constexpr int kHcComb = kHc * kHc;

// One site's weights (FP32 device copies) and their gradients (+= when not null: each is optional)
struct MhcSite {
	int hidden = 0;
	const float* fn = nullptr;            // [kHcMix, kHc * hidden]
	const float* base = nullptr;          // [kHcMix]
	const float* scale = nullptr;         // [3]: pre, post, comb
	const float* norm_w = nullptr;        // [hidden]: the sublayer's input RMSNorm weight, used as is (no 1 + w)
	float* gfn = nullptr;
	float* gbase = nullptr;
	float* gscale = nullptr;
	float* gnorm = nullptr;
	int iters = 20;                       // Sinkhorn rounds: one column normalization, then iters - 1 x ( row, column )
	float rms_eps = 1e-5f, hc_eps = 1e-6f, norm_eps = 1e-5f;
};

// What the pre step's backward reads, per row: mix [rows, kHcMix] (the logits before rmr, as the inference's mix_logits_buf), rmr [rows],
// pre [rows, kHc], inv [rows] (the sublayer norm's 1 / sqrt( mean( u^2 ) + norm_eps )); u itself is recomputed from X and pre
struct MhcCache {
	float* mix = nullptr;
	float* rmr = nullptr;
	float* pre = nullptr;
	float* inv = nullptr;
};
constexpr int kMhcCacheFloats = kHcMix + 1 + kHc + 1;   // per row
// The cache's buffers carved from buf (rows * kMhcCacheFloats floats)
MhcCache mhc_cache( float* buf, int rows );

// The pre step: the sublayer input a [rows, hidden], post [rows, kHc] and comb [rows, kHcComb] (the post step's inputs), the cache filled
void mhc_pre_forward( Blas& b, const MhcSite& s, const float* X, int rows, const MhcCache& c, float* a, float* post, float* comb );
// Its backward from da [rows, hidden], dpost [rows, kHc] and dcomb [rows, kHcComb] (dpost, dcomb null: zero) at the same X and cache: dX +=
// the gradient through all three (each depends on X through l and rmr, a also through u and pre), and the site's gradients that are wanted.
// dX must not alias X; scratch: mhc_pre_scratch_floats
void mhc_pre_backward( Blas& b, const MhcSite& s, const float* X, int rows, const MhcCache& c, const float* da, const float* dpost, const float* dcomb,
	float* dX, float* scratch );
// rows * ( 2 + kHc + 2 kHcMix + ( 2 iters - 1 ) kHcComb ) (the Sinkhorn's iterates: 624 floats a row at 20 rounds)
size_t mhc_pre_scratch_floats( const MhcSite& s, int rows );

// The post step X_new_h = post_h y + sum_g comb[g][h] X_old_g (X_new may alias X_old: a column's four values are read before they are written,
// but the backward needs X_old)
void mhc_post_forward( const float* X_old, const float* post, const float* comb, const float* y, int rows, int hidden, float* X_new,
	backend::stream_t stream );
// Its backward from dX_new: dX_old += comb^T dX_new per column; dy [rows, hidden], dpost [rows, kHc] and dcomb [rows, kHcComb] overwritten
// (dcomb[g][h] = dX_new_h . X_old_g, dpost_h = dX_new_h . y). dX_old must not alias dX_new
void mhc_post_backward( const float* X_old, const float* post, const float* comb, const float* y, const float* dX_new, int rows, int hidden, float* dX_old,
	float* dy, float* dpost, float* dcomb, backend::stream_t stream );

// The head's stream mean h = sum_h X_h / kHc [rows, hidden] (the final RMSNorm after it is the generic one); backward dX_h += dh / kHc
void hc_mean_forward( const float* X, int rows, int hidden, float* h, backend::stream_t stream );
void hc_mean_backward( const float* dh, int rows, int hidden, float* dX, backend::stream_t stream );

} // namespace engine::train::glm5_next
