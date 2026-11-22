#pragma once

#include "engine/config.h"
#include "engine/backend/types.h"

namespace engine::modules {

// Per-row statistics of full-vocabulary logits on the device, for the perplexity test and its teacher files (test_ppl): what a host
// softmax over each row gave, without copying the rows back
struct LogitRows {
	static constexpr int MAX_K = 256;   // top entries per row at most
	// logits [rows, vocab] FP16. Per row r: lse[r] = log sum exp (double), top[r] the argmax (the first of equal maxima); with target (device,
	// [rows], -1: none) nll[r] = lse[r] - logit[target[r]]; with k > 0 the k largest logits (larger first, ties to the lower id) as ids [rows, k]
	// with their probabilities p [rows, k] and tail[r] = max( 0, 1 - sum p ); with a teacher (t_ids, t_p [rows, tk], t_tail [rows]) kl[r] =
	// KL( teacher || softmax ) over its tk entries and its lumped tail (the model's mass outside them, at least 1e-12). Null outputs are skipped
	static void score( const backend::half_t* logits, int rows, int vocab, const int* target, int k, int* ids, float* p, float* tail, const int* t_ids,
		const float* t_p, const float* t_tail, int tk, double* lse, float* nll, int* top, float* kl, backend::stream_t stream );
};

} // namespace engine::modules
