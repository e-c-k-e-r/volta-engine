#pragma once

#include "engine/runtime/speculator.h"

namespace engine::runtime {

class ModelCluster;

// Speculation from the DSpark block drafter that ships with DeepSeek-V4.1 (three small blocks reading the trunk's stream-mean taps; the model keeps
// the taps and the drafter's window KV, see ModelPipeline::dspark_forward). Each round drafts a block of tokens from the last committed token,
// keeps the prefix whose confidence head says it is worth verifying (option dspark_conf), verifies it in one target forward and accepts the
// longest matching prefix. A draft offered from outside (spec_suffix) replaces drafting. A round without a draft is a plain step.
class DSparkRunner : public Speculator {
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
	void on_prefill_chunk( int slot, const int* tokens, int rows, int pos0 ) override;
	// Snapshots carry the last WINDOW positions' taps: a restore re-derives the drafter's window KV from them
	bool needs_taps() const override { return true; }
	bool cheap_snapshot_taps() const override { return true; }
	void on_restore( int slot, int tokens ) override;
	void step_stats( int* attempts, int* hits ) const override;
	void profile_dump() override;

private:
	ModelCluster* mc_ = nullptr;
	static constexpr int kMaxSlots = 8;
	struct SlotState {
		bool owned = false;
		bool fresh = true;     // no prefill chunk seen since the claim
		int valid_from = 0;    // first position whose taps were recorded (a prefill resumed from a restored prefix starts there)
		int ingested = 0;      // the drafter's window KV holds the positions below this
		int n = 0;             // tokens offered for the next round
		int tok[8] = {};
	};
	SlotState st_[kMaxSlots];
	int block_ = 0;
	float conf_th_ = 0.5f;
	int att_[8] = {};
	int hit_[8] = {};
	// profile
	long rounds_ = 0, drafted_rounds_ = 0, offered_rounds_ = 0, proposed_ = 0, accepted_ = 0;
	double draft_ms_ = 0.0, verify_ms_ = 0.0;
};

} // namespace engine::runtime
