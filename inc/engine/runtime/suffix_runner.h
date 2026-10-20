#pragma once

#include "engine/runtime/speculator.h"

namespace engine::runtime {

class ModelCluster;

// Speculation without a drafter: the draft is what followed the last earlier occurrence of the request's current text suffix (offered by the
// generator, option spec_suffix), verified in one target forward; the longest matching prefix is accepted. A round without an offer is a plain
// step. For models whose caches are indexed by position, so that a rejected draft needs no state repair (ModelPipeline::free_rollback).
class SuffixRunner : public Speculator {
public:
	bool init( ModelCluster& mc );
	void shutdown() override { mc_ = nullptr; }

	void claim( int slot ) override;
	void release( int slot ) override;
	bool owns( int slot ) const override { return slot >= 0 && slot < kMaxSlots && st_[slot].owned; }
	bool per_slot() const override { return true; }
	SpecResult speculate( int slot, int anchor, int pos, float depth_threshold, const SamplingParams* sp, Rng* rng ) override;
	void offer_draft( int slot, const int* tokens, int n ) override;
	bool takes_offered_draft() const override { return true; }
	bool needs_taps() const override { return false; }
	void step_stats( int* attempts, int* hits ) const override;

private:
	ModelCluster* mc_ = nullptr;
	static constexpr int kMaxSlots = 8;
	struct SlotState {
		bool owned = false;
		int n = 0;           // tokens offered for the next round
		int tok[8] = {};
	};
	SlotState st_[kMaxSlots];
	int att_[8] = {};
	int hit_[8] = {};
};

} // namespace engine::runtime
