#pragma once

#include "engine/runtime/speculator.h"

namespace engine::runtime {

class ModelCluster;

// Speculation from the checkpoint's own multi-token-prediction head. Each round: catch the head up over the rows the last verify
// committed (their true hidden states), draft `depth` tokens by chaining the head on its own output, verify them in one target
// forward as DFlash2 does, accept the longest matching prefix.
//
// The head keeps a KV layer per slot. Position i's row is (target hidden at i, token i + 1) -> prediction of token i + 2. The hidden of
// the rows a verify committed is kept (the "pending" rows) until the next round supplies the token that follows the last of them.
class MtpRunner : public Speculator {
public:
	bool init( ModelCluster& mc );
	void shutdown() override { mc_ = nullptr; owner_ = -1; }

	void claim( int slot ) override;
	void release( int slot ) override;
	bool owns( int slot ) const override { return owner_ == slot; }
	SpecResult speculate( int slot, int anchor, int pos, float depth_threshold, const SamplingParams* sp, Rng* rng ) override;
	void on_prefill_chunk( int slot, const int* tokens, int rows, int pos0 ) override;
	void on_dense_row( int slot, int token, int pos, int row ) override;
	bool needs_taps() const override { return false; }
	void step_stats( int* attempts, int* hits ) const override;

private:
	ModelCluster* mc_ = nullptr;
	int owner_ = -1;
	int depth_ = 3;
	int hist_base_ = 0;  // the model keeps two halves of saved hidden rows; steps read one while the next pending rows go to the other
	int pend_pos_ = 0;   // position of the first pending row
	int pend_n_ = 0;     // pending rows (hidden kept in the model); 0 = none (cold start or after a restore)
	int pend_tok_[8] = {};   // tokens following pending rows 0 .. pend_n_ - 2
	float cond_[8] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };   // survival EMA of draft step k given steps < k were accepted
	int att_[8] = {};
	int hit_[8] = {};
};

} // namespace engine::runtime
