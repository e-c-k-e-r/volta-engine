#pragma once

#include "engine/models/model.h"
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
	void shutdown() override { mc_ = nullptr; }

	void claim( int slot ) override;
	void release( int slot ) override;
	bool owns( int slot ) const override { return slot >= 0 && slot < kMaxSlots && st_[slot].owned; }
	bool per_slot() const override { return true; }
	SpecResult speculate( int slot, int anchor, int pos, float depth_threshold, const SamplingParams* sp, Rng* rng ) override;
	bool joint() const override;
	void speculate_joint( int n, const int* slots, const int* anchors, const int* poss, const SamplingParams* const* sps, Rng* const* rngs, SpecResult* out ) override;
	void on_prefill_chunk( int slot, const int* tokens, int rows, int pos0 ) override;
	void on_dense_row( int slot, int token, int pos, int row ) override;
	bool needs_taps() const override { return false; }
	void step_stats( int* attempts, int* hits ) const override;

private:
	ModelCluster* mc_ = nullptr;
	static constexpr int kMaxSlots = 8;
	static int hist_off( int slot ) { return slot * 2 * ( models::kDraftM + 1 ); }
	// Per slot: the saved hidden rows live in the slot's region of the model's history buffer (two halves of kDraftM + 1 rows; steps read one
	// while the next pending rows go to the other)
	struct SlotState {
		bool owned = false;
		int hist_base = 0;
		int pend_pos = 0;    // position of the first pending row
		int pend_n = 0;      // pending rows (hidden kept in the model); 0 = none (cold start or after a restore)
		int pend_tok[8] = {};   // tokens following pending rows 0 .. pend_n - 2
	};
	SlotState st_[kMaxSlots];
	int depth_ = 3;
	float cond_[8] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };   // survival EMA of draft step k given steps < k were accepted
	int att_[8] = {};
	int hit_[8] = {};
};

} // namespace engine::runtime
